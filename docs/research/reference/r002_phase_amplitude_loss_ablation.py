"""Bounded NumPy shared-phase Adam diagnostic for max-vs-mean AMPLITUDE losses (v1.1).
One fixed synthetic quartet, official 128 grid/four z, 3 seeds, 100 steps each.
NOT neural training, NOT production code or a general choice of best loss.
"""
import numpy as np,json,platform
N=128;P=4;n=N*N;lam=532e-9;dx=10e-6
zs=np.array([.0241,.0253,.0265,.0277])[:,None,None]
v=(np.arange(N)-N//2)*dx
Y,X=np.meshgrid(v,v,indexing="ij")
qin=np.exp(1j*np.pi*(X*X+Y*Y)/(lam*zs))
stepsout=lam*zs/(N*dx)
ii=np.arange(N)-N//2
XO=ii[None,None,:]*stepsout;YO=ii[None,:,None]*stepsout
qout=(N*dx*dx)/(1j*lam*zs)*np.exp(2j*np.pi*zs/lam+1j*np.pi*(XO*XO+YO*YO)/(lam*zs))
def fc(u):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u,axes=(-2,-1)),axes=(-2,-1),norm="ortho"),axes=(-2,-1))
def hc(u):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u,axes=(-2,-1)),axes=(-2,-1),norm="ortho"),axes=(-2,-1))
y,x=np.meshgrid(np.linspace(-1,1,N),np.linspace(-1,1,N),indexing="ij")
ts=[
 np.exp(-((x+.25)**2+(y+.10)**2)/.10),
 np.exp(-((np.sqrt(x*x+y*y)-.45)/.10)**2),
 np.exp(-((x-.25)**2+(y+.25)**2)/.08)+.7*np.exp(-((x+.35)**2+(y-.30)**2)/.06),
 ((abs(x)<.10)&(abs(y)<.65)).astype(float)+.6*((abs(y)<.08)&(abs(x)<.65)).astype(float)]
target=np.stack([.02+.98*t/t.max() for t in ts])
def forward(phi):
    return qout*fc(qin*np.exp(1j*phi)[None,:,:])
def objective(phi,kind,need_grad=True):
    out=forward(phi);I=abs(out)
    if np.any(I<=1e-16):raise ValueError('Zero amplitude in derivative diagnostic')
    if kind=="max":
        s=I.max(axis=(-2,-1),keepdims=True);J=I/s;e=J-target
        d=2*e/(P*n*s)
        bonus=2*np.sum(e*I,axis=(-2,-1),keepdims=True)/(P*n*s*s)
        for p in range(P):d[p].flat[np.argmax(I[p])]-=bonus[p,0,0]
    else:
        s=I.mean(axis=(-2,-1),keepdims=True);J=I/s
        e=J-target/target.mean(axis=(-2,-1),keepdims=True)
        d=2/(P*n*s)*(e-np.mean(e*J,axis=(-2,-1),keepdims=True))
    L=float(np.mean(e*e))
    if not need_grad:return L
    dI=d/(2*I)
    back=np.sum(np.conj(qin)*hc(np.conj(qout)*dI*out),axis=0)
    grad=2*np.real(np.conj(back)*(1j*np.exp(1j*phi)))
    return L,grad
def evaluate(phi):
    I=abs(forward(phi));display=I/I.max(axis=(-2,-1),keepdims=True)
    metrics=[]
    for a,b in zip(display,target):
        ac=a-a.mean();bc=b-b.mean()
        corr=np.sum(ac*bc)/np.sqrt(np.sum(ac*ac)*np.sum(bc*bc))
        va=np.mean(ac*ac);vb=np.mean(bc*bc);cov=np.mean(ac*bc)
        ssim=(2*a.mean()*b.mean()+.01**2)*(2*cov+.03**2)/((a.mean()**2+b.mean()**2+.01**2)*(va+vb+.03**2))
        metrics.append({"max_display_mse":float(np.mean((a-b)**2)),"pearson":float(corr),"global_ssim":float(ssim)})
    power=np.sum(I*I,axis=(-2,-1))*stepsout[:,0,0]**2
    return {"per_plane":metrics,"mean_display_mse":float(np.mean((display-target)**2)),
            "mean_pearson":float(np.mean([m["pearson"] for m in metrics])),"mean_global_ssim":float(np.mean([m["global_ssim"] for m in metrics])),
            "max_area_power_relative_error":float(np.max(abs(power-n*dx*dx))/(n*dx*dx))}
checkphi=np.random.default_rng(7).uniform(-np.pi,np.pi,(N,N))
gradchecks=[]
for kind in ("max","mean"):
    L,g=objective(checkphi,kind)
    for r,c in [(0,0),(45,69),(127,127)]:
        plus=checkphi.copy();minus=checkphi.copy();plus[r,c]+=1e-5;minus[r,c]-=1e-5
        fd=(objective(plus,kind,False)-objective(minus,kind,False))/2e-5
        err=abs(fd-g[r,c]);assert err<1e-7
        gradchecks.append({"loss":kind,"pixel":[r,c],"absolute_error":float(err)})
rows=[]
for seed in [0,1,2]:
    init=np.random.default_rng(seed).uniform(-np.pi,np.pi,(N,N))
    initial=evaluate(init)
    for kind in ("max","mean"):
        phi=init.copy();m=np.zeros_like(phi);v=np.zeros_like(phi)
        best=phi.copy();bestL=objective(phi,kind,False);startL=bestL
        for step in range(1,101):
            L,g=objective(phi,kind);assert np.isfinite(g).all()
            m=.9*m+.1*g;v=.999*v+.001*g*g
            phi-=.05*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
            L=objective(phi,kind,False)
            if L<bestL:bestL=L;best=phi.copy()
        rows.append({"seed":seed,"loss":kind,"steps":100,"lr":.05,"initial_own_objective":startL,"best_own_objective":bestL,"initial_eval":initial,"best_eval":evaluate(best)})
print(json.dumps({"scope":"One synthetic positive quartet; continuous phase; own NumPy Adam/analytic adjoint validated by central differences. AMPLITUDE domain, same max-amplitude-display evaluation for both losses, PR15 head301ea47 proposal. Not neural training, PNG8 benchmark, actual team code or general ranking.",
"environment":{"python":platform.python_version(),"numpy":np.__version__},"gradient_checks":gradchecks,"runs":rows},indent=2))
