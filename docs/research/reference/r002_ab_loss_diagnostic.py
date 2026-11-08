"""Coordinator diagnostic: analytic intensity gradients, NOT a PyTorch/Fresnel training run."""
import json, platform
import numpy as np
n=128*128
x=np.ones(n,dtype=np.float64); x[n//2]=100.
t=np.full(n,.5,dtype=np.float64)
def value_grad(x,kind):
    if kind=="max":
        s=x.max();j=x/s;e=j-t
        g=2*e/(n*s)
        g[x.argmax()]-=2*np.dot(e,x)/(n*s*s)
    else:
        s=x.mean();j=x/s;e=j-t/t.mean()
        g=2/(n*s)*(e-np.mean(e*j))
    return float(np.mean(e*e)),g
results={}
for kind in ("max","mean"):
    loss,g=value_grad(x,kind)
    p=n//2;fd=[]
    for k in (0,p):
        xp=x.copy();xm=x.copy()
        xp[k]+=1e-3;xm[k]-=1e-3
        finite=(value_grad(xp,kind)[0]-value_grad(xm,kind)[0])/2e-3
        error=abs(finite-g[k])
        assert error<1e-9
        fd.append({"index":k,"analytic":float(g[k]),"finite_difference":finite,"absolute_error":float(error)})
    results[kind]={"loss":loss,"peak_over_mean_abs_all":float(abs(g[p])/np.abs(g).mean()),"peak_over_background_abs":float(abs(g[p])/abs(g[0])),"finite_difference":fd}
assert abs(results["max"]["peak_over_mean_abs_all"]-results["mean"]["peak_over_mean_abs_all"])<1e-9
results["scope"]="Coordinator NumPy analytic-gradient check for Gemini A's unconstrained intensity fixture; not Gemini execution, not literal PyTorch, not phase/neural/Fresnel benchmark."
results["environment"]={"python":platform.python_version(),"numpy":np.__version__}
print(json.dumps(results,indent=2))
