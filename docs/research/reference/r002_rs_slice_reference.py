"""Bounded RS-I vs continuous Fresnel diagnostic on full-fill piecewise constant 128x128 cells.
Three output x samples at y=0 per z; not a full 2D/ROI error bound.
Global carrier exp(ikz) removed from BOTH fields for stable comparison.
"""
import numpy as np,json,platform
N=128;d=10e-6;lam=532e-9;k=2*np.pi/lam
us=(np.arange(N)-N//2)*d
Yc,Xc=np.meshgrid(us,us,indexing="ij")
fields={"uniform_lens_f25.3":np.exp(-1j*np.pi*(Xc*Xc+Yc*Yc)/(lam*.0253)),
        "uniform_tilt35_per_mm":np.exp(2j*np.pi*35000*Xc)}
rows=[]
for z in [.0241,.0253,.0265,.0277]:
    xo=np.array([0,20,52])*lam*z/(N*d)
    values={name:{} for name in fields}
    for q in [2,4,8]:
        nodes,w=np.polynomial.legendre.leggauss(q)
        src=(us[:,None]+d/2*nodes[None,:]).ravel()
        sy,sx=np.meshgrid(src,src,indexing="ij")
        w2=np.outer(np.tile(w,N),np.tile(w,N))*d*d/4
        expanded={name:np.repeat(np.repeat(f,q,axis=0),q,axis=1) for name,f in fields.items()}
        arrays={name:{"f":[],"rs":[]} for name in fields}
        for xx in xo:
            rho2=(xx-sx)**2+sy*sy
            R=np.sqrt(z*z+rho2)
            fr=np.exp(1j*np.pi*rho2/(lam*z))/(1j*lam*z)
            rs=z/(2*np.pi*R*R)*(1/R-1j*k)*np.exp(1j*k*rho2/(R+z))
            for name in fields:
                arrays[name]["f"].append(np.sum(w2*expanded[name]*fr))
                arrays[name]["rs"].append(np.sum(w2*expanded[name]*rs))
        for name in fields:values[name][q]={key:np.asarray(v) for key,v in arrays[name].items()}
    def rel(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))
    for name in fields:
        a=values[name]
        rows.append({"field":name,"z_m":z,"output_x_m":xo.tolist(),
            "fresnel4_to8_complex_relative_l2":rel(a[4]["f"],a[8]["f"]),
            "rs4_to8_complex_relative_l2":rel(a[4]["rs"],a[8]["rs"]),
            "fresnel2_to4_complex_relative_l2":rel(a[2]["f"],a[4]["f"]),
            "rs2_to4_complex_relative_l2":rel(a[2]["rs"],a[4]["rs"]),
            "fresnel_vs_rs_complex_relative_l2_3points":rel(a[8]["f"],a[8]["rs"]),
            "fresnel_vs_rs_intensity_relative_l2_3points":rel(abs(a[8]["f"])**2,abs(a[8]["rs"])**2)})
print(json.dumps({"scope":"Only 3 output x points, y=0; two pixel-held fields; all 4 z; no full ROI or worst-case bound; finite Gauss convergence only; scalar RS-I model, no real optical calibration.",
"environment":{"python":platform.python_version(),"numpy":np.__version__},"cases":rows},indent=2))
