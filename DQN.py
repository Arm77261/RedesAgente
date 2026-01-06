import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from collections import deque

# Device y tipo de dato
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

# Parámetros de imagen y red
IMG_SIZE = 224
STRIDE = 16
Z_DIM = 256

# Parámetros de entrenamiento
MAX_STEPS = 2000
EPISODES = 400
BATCH_SIZE = 64

# Hiperparámetros RL
GAMMA = 0.95
LR = 1e-4
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

# Recompensas
FEAT_MIN = 0.90
MIN_POINTS = 70
OUT_PENALTY = -1.5
POINT_REWARD = 1.0
SUCCESS_REWARD = 15.0

# Rutas de datos
TRAIN_DIR = "trains"
VAL_DIR   = "validations1"
VAE_PATH  = "vae_models/vae_features.pth"
NUM_CLASSES = len(os.listdir(TRAIN_DIR))

# Prior espacial
SPATIAL_PRIOR_PATH = os.path.expanduser("~/spatial_prior.pt")
SPATIAL_PRIOR = torch.zeros((NUM_CLASSES, IMG_SIZE, IMG_SIZE), device=DEVICE, dtype=torch.float32)
PRIOR_DECAY = 0.995  

# Normalizar prior
def get_normalized_prior(class_id: int):
    prior = SPATIAL_PRIOR[class_id]
    if prior.max() > 0:
        prior = prior / (prior.max() + 1e-6)
    return prior

# Guardar prior
def save_spatial_prior(path: str = SPATIAL_PRIOR_PATH):
    torch.save(SPATIAL_PRIOR.detach().cpu(), path)
    print(f"Spatial prior guardado → {path}")

# Cargar prior
def load_spatial_prior(path: str = SPATIAL_PRIOR_PATH):
    global SPATIAL_PRIOR
    if not isinstance(path, (str, bytes, os.PathLike)):
        SPATIAL_PRIOR.zero_()
        return
    if os.path.exists(path):
        prior = torch.load(path, map_location=DEVICE)
        if isinstance(prior, torch.Tensor) and prior.shape == SPATIAL_PRIOR.shape:
            SPATIAL_PRIOR[:] = prior.to(DEVICE)
        else:
            SPATIAL_PRIOR.zero_()
    else:
        SPATIAL_PRIOR.zero_()

# Mostrar prior
def show_spatial_prior(class_id, title=None):
    prior = get_normalized_prior(class_id).cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(prior, cmap="hot")
    plt.title(title or f"Spatial Prior clase {class_id}")
    plt.colorbar()
    plt.axis("off")
    plt.show()

# Transformación de imagen
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Bloque residual
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch)
        )
    def forward(self, x):
        return F.relu(x + self.block(x))

# VAE de features
class FeatureVAE(nn.Module):
    def __init__(self, z=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,64,3,2,1), nn.BatchNorm2d(64), nn.ReLU(), ResBlock(64),
            nn.Conv2d(64,128,3,2,1), nn.BatchNorm2d(128), nn.ReLU(), ResBlock(128),
            nn.Conv2d(128,256,3,2,1), nn.BatchNorm2d(256), nn.ReLU(), ResBlock(256),
            nn.Conv2d(256,512,3,2,1), nn.BatchNorm2d(512), nn.ReLU(), ResBlock(512),
            nn.Conv2d(512,512,3,2,1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.fc_mu  = nn.Linear(512*7*7, z)
        self.fc_dec = nn.Linear(z, 512*7*7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512,512,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(512,256,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,3,4,2,1),
            nn.Sigmoid()
        )
    def reconstruct(self, x):
        h = self.enc(x).flatten(1)
        z = self.fc_mu(h)
        h = self.fc_dec(z).view(-1,512,7,7)
        return self.dec(h)

# Cargar VAE
def load_vae():
    ckpt = torch.load(VAE_PATH, map_location=DEVICE)
    enc_only = {k: v for k, v in ckpt.items() if k.startswith("enc.") or k.startswith("fc_mu.")}
    vae = FeatureVAE().to(DEVICE)
    vae.load_state_dict(enc_only, strict=False)
    vae.eval()
    return vae

# Heatmap perceptual
def heatmap_from_vae(img_t, vae):
    with torch.no_grad():
        recon = vae.reconstruct(img_t.unsqueeze(0))
        if recon.shape[-1] != IMG_SIZE:
            recon = F.interpolate(recon, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        err = (recon - img_t.unsqueeze(0)).pow(2).mean(1)
        heat = torch.exp(-err)
        heat /= heat.max() + 1e-6
    return heat.squeeze().cpu().numpy()

# BBox de puntos
def compute_bbox(points, margin=12):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(0, min(xs)-margin), max(0, min(ys)-margin), min(IMG_SIZE-1, max(xs)+margin), min(IMG_SIZE-1, max(ys)+margin))

# Centro de puntos
def compute_centroid(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int(np.mean(xs)), int(np.mean(ys))

# Red DQN
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6,256), nn.ReLU(inplace=True), nn.Linear(256,4))
    def forward(self, x):
        return self.net(x)

# Entorno de localización
class LocateEnv:
    def __init__(self, img, vae_feat, vae_disc, target_class):
        assert target_class is not None
        self.img = img
        self.img_t = transform(img).to(DEVICE)
        self.vae_feat = vae_feat
        self.vae_disc = vae_disc
        self.target_class = target_class
        self.reset()
    def reset(self):
        self.heat_feat = heatmap_from_vae(self.img_t, self.vae_feat)  # heatmap VAE
        heat_sem = animal_heatmap(self.img_t, self.vae_disc, self.target_class)  # heatmap semántico
        prior = get_normalized_prior(self.target_class).cpu().numpy()  # prior
        self.heat_animal = 0.85*heat_sem + 0.15*prior
        self.heat_animal -= self.heat_animal.min()
        self.heat_animal /= self.heat_animal.max() + 1e-6
        thr = np.quantile(self.heat_feat, 0.9)
        ys, xs = np.where(self.heat_feat > thr)
        if len(xs)==0: self.x=self.y=IMG_SIZE//2
        else:
            i=random.randint(0,len(xs)-1)
            self.x, self.y = int(xs[i]), int(ys[i])
        self.visited=set()
        self.rejected=set()
        self.path=[]
        self.steps=0
        self.bbox=(0,0,IMG_SIZE-1,IMG_SIZE-1)
        return self._state()
    def _state(self):
        animal_score = self.heat_animal[self.y,self.x]
        gx = self.heat_animal[self.y,min(self.x+1,IMG_SIZE-1)]-self.heat_animal[self.y,max(self.x-1,0)]
        gy = self.heat_animal[min(self.y+1,IMG_SIZE-1),self.x]-self.heat_animal[max(self.y-1,0),self.x]
        return torch.tensor([self.x/IMG_SIZE, self.y/IMG_SIZE, len(self.visited)/MIN_POINTS, animal_score, gx, gy], device=DEVICE, dtype=DTYPE)
    def step(self, action):
        self.steps +=1
        if random.random()<0.05:
            self.x=random.randint(0,IMG_SIZE-1)
            self.y=random.randint(0,IMG_SIZE-1)
        else:
            if action==0: self.y-=STRIDE
            elif action==1: self.y+=STRIDE
            elif action==2: self.x-=STRIDE
            elif action==3: self.x+=STRIDE
        self.x=int(np.clip(self.x,0,IMG_SIZE-1))
        self.y=int(np.clip(self.y,0,IMG_SIZE-1))
        key=(self.x,self.y)
        self.path.append(key)
        animal_score=self.heat_animal[self.y,self.x]
        if animal_score<0.3:
            self.rejected.add(key)
            return self._state(), OUT_PENALTY, False
        reward=POINT_REWARD*animal_score
        if key in self.visited: reward-=0.1
        else: self.visited.add(key)
        local_density=sum(1 for (x,y) in self.visited if abs(x-self.x)<STRIDE and abs(y-self.y)<STRIDE)
        reward-=0.02*local_density
        if len(self.visited)>=MIN_POINTS:
            scores=[self.heat_animal[y,x] for (x,y) in self.visited]
            xs=[x for (x,_) in self.visited]
            ys=[y for (_,y) in self.visited]
            if np.mean(scores)>0.6 and np.std(xs)<IMG_SIZE*0.15 and np.std(ys)<IMG_SIZE*0.15:
                SPATIAL_PRIOR[self.target_class].mul_(PRIOR_DECAY)
                for (x,y) in self.visited: SPATIAL_PRIOR[self.target_class,y,x]+=0.2
                SPATIAL_PRIOR[self.target_class].clamp_(0,50)
            self.x,self.y=compute_centroid(self.visited)
            self.bbox=compute_bbox(self.visited)
            reward+=SUCCESS_REWARD
            return self._state(), reward, True
        if self.steps>=MAX_STEPS: return self._state(), -1.0, True
        return self._state(), reward, False

# Discriminative VAE
class DiscriminativeVAE(nn.Module):
    def __init__(self,num_classes,z=Z_DIM):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv2d(3,64,3,2,1), nn.BatchNorm2d(64), nn.ReLU(), ResBlock(64),
            nn.Conv2d(64,128,3,2,1), nn.BatchNorm2d(128), nn.ReLU(), ResBlock(128),
            nn.Conv2d(128,256,3,2,1), nn.BatchNorm2d(256), nn.ReLU(), ResBlock(256),
            nn.Conv2d(256,512,3,2,1), nn.BatchNorm2d(512), nn.ReLU(), ResBlock(512),
            nn.Conv2d(512,512,3,2,1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.fc_mu=nn.Linear(512*7*7,z)
        self.fc_logvar=nn.Linear(512*7*7,z)
        self.classifier=nn.Sequential(nn.Linear(z,256),nn.ReLU(inplace=True),nn.Linear(256,num_classes))
        self.fc_dec=nn.Linear(z,512*7*7)
        self.dec=nn.Sequential(
            nn.ConvTranspose2d(512,512,4,2,1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64,3,4,2,1), nn.Sigmoid()
        )
    def encode(self,x): h=self.enc(x).flatten(1); return self.fc_mu(h), self.fc_logvar(h)
    def reparam(self,mu,logvar): std=torch.exp(0.5*logvar); return mu+torch.randn_like(std)*std
    def decode(self,z): return self.dec(self.fc_dec(z).view(-1,512,7,7))
    def forward(self,x): mu,logvar=self.encode(x); z=self.reparam(mu,logvar); recon=self.decode(z); logits=self.classifier(mu); return recon,mu,logvar,logits

# Heatmap semántico
def animal_heatmap(img_t, vae_disc, target_class):
    with torch.no_grad():
        x=img_t.unsqueeze(0)
        feats=vae_disc.enc(x).mean(1,keepdim=True)
        feats=F.interpolate(feats,size=(IMG_SIZE,IMG_SIZE),mode="bilinear",align_corners=False)
        heat=(feats.squeeze()-feats.min())/(feats.max()-feats.min()+1e-6)
    return heat.cpu().numpy()



    

def train():

    load_spatial_prior()

    vae_feat = load_vae().eval()

    classes = sorted(os.listdir(TRAIN_DIR))
    class_to_id = {c:i for i,c in enumerate(classes)}

    vae_disc = DiscriminativeVAE(num_classes=len(classes)).to(DEVICE)
    vae_disc.load_state_dict(
        torch.load("vae_models/vae_discriminative.pth", map_location=DEVICE)
    )
    vae_disc.eval()

    qnet   = DQN().to(DEVICE)
    target = DQN().to(DEVICE)
    target.load_state_dict(qnet.state_dict())
    target.eval()

    opt = torch.optim.Adam(qnet.parameters(), lr=LR)
    memory = deque(maxlen=70000)


    images = []
    for cls in classes:
        p = os.path.join(TRAIN_DIR, cls)
        for f in os.listdir(p):
            images.append((os.path.join(p, f), class_to_id[cls]))

    eps = EPS_START

    DEBUG_EP_EVERY   = 5
    DEBUG_STEP_EVERY = 40

    
    for ep in range(EPISODES):

        img_path, cls_id = random.choice(images)
        cls_name = classes[cls_id]

        img = Image.open(img_path).convert("RGB")

        env = LocateEnv(
            img=img,
            vae_feat=vae_feat,
            vae_disc=vae_disc,
            target_class=cls_id
        )

        state = env.reset()
        total_reward = 0.0

        for step in range(MAX_STEPS):

            # ε-greedy
            if random.random() < eps:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    action = qnet(state).argmax().item()

            next_state, reward, done = env.step(action)

            memory.append((
                state.detach(),
                action,
                reward,
                next_state.detach()
            ))

            state = next_state
            total_reward += reward

       
            if len(memory) >= BATCH_SIZE:
                S, A, R, SN = zip(*random.sample(memory, BATCH_SIZE))

                S  = torch.stack(S)
                SN = torch.stack(SN)
                A  = torch.tensor(A, device=DEVICE).unsqueeze(1)
                R  = torch.tensor(R, device=DEVICE)

                q_pred = qnet(S).gather(1, A).squeeze()
                with torch.no_grad():
                    q_next = target(SN).max(1)[0]

                loss = F.smooth_l1_loss(q_pred, R + GAMMA * q_next)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(qnet.parameters(), 1.0)
                opt.step()

           
            if ep % DEBUG_EP_EVERY == 0 and step % DEBUG_STEP_EVERY == 0:

                plt.figure(figsize=(6,6))
                plt.imshow(env.heat_animal, cmap="hot", alpha=0.7)

                if env.rejected:
                    rx, ry = zip(*env.rejected)
                    plt.scatter(rx, ry, c="red", s=8, label="rechazado")

                if env.visited:
                    vx, vy = zip(*env.visited)
                    plt.scatter(vx, vy, c="lime", s=25, label="aceptado")

                if env.path:
                    px, py = env.path[-1]
                    plt.scatter(px, py, c="cyan", s=90, marker="x", label="actual")

                if len(env.visited) >= 5:
                    x1, y1, x2, y2 = env.bbox
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            edgecolor="white",
                            facecolor="none",
                            linewidth=2
                        )
                    )

                plt.title(
                    f"Ep {ep} | {cls_name} | step {step} | ε={eps:.2f}"
                )
                plt.axis("off")
                plt.legend(loc="lower right")
                plt.show()

            if done:
                break

       
        if ep % 5 == 0:
            target.load_state_dict(qnet.state_dict())

        eps = max(EPS_END, eps * EPS_DECAY)

        print(
            f" Ep {ep:03d} | Clase {cls_name:<10} | "
            f"Reward {total_reward:7.2f} | ε={eps:.3f}"
        )

   
    torch.save(qnet.state_dict(), "dqn_mark_animal.pth")
    save_spatial_prior()

    print("DQN + Spatial Prior guardados")



def validate():

    load_spatial_prior()

    vae_feat = load_vae().eval()

    classes = sorted(os.listdir(TRAIN_DIR))

    vae_disc = DiscriminativeVAE(num_classes=len(classes)).to(DEVICE)
    vae_disc.load_state_dict(
        torch.load("vae_models/vae_discriminative.pth", map_location=DEVICE)
    )
    vae_disc.eval()

    qnet = DQN().to(DEVICE)
    qnet.load_state_dict(torch.load("dqn_mark_animal.pth", map_location=DEVICE))
    qnet.eval()

    for f in random.sample(os.listdir(VAL_DIR), min(15, len(os.listdir(VAL_DIR)))):

        img = Image.open(os.path.join(VAL_DIR, f)).convert("RGB")
        W, H = img.size

        
        with torch.no_grad():
            x = transform(img).unsqueeze(0).to(DEVICE)
            _, mu, _, logits = vae_disc(x)
            probs = F.softmax(logits, dim=1)
            cls_id = probs.argmax(1).item()
            cls_name = classes[cls_id]

        env = LocateEnv(
            img=img,
            vae_feat=vae_feat,
            vae_disc=vae_disc,
            target_class=cls_id
        )

        state = env.reset()

        for _ in range(MAX_STEPS):
            with torch.no_grad():
                state, _, done = env.step(qnet(state).argmax().item())
            if done:
                break

        
        points = list(env.visited)
        if len(points) > 30:
            points = random.sample(points, 30)

        sx, sy = W / IMG_SIZE, H / IMG_SIZE

        plt.figure(figsize=(6,6))
        plt.imshow(img)

        if points:
            xs, ys = zip(*points)
            xs = [x * sx for x in xs]
            ys = [y * sy for y in ys]
            plt.scatter(xs, ys, c="cyan", s=40)

        plt.title(f"Detectado: {cls_name}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    train()
    validate()
