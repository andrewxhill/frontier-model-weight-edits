"""Direct closed-form numerical edits. No gradients, optimiser or test inputs."""
import numpy as np

def ridge(X,Y,lam,weights=None):
 X=np.asarray(X,np.float64);Y=np.asarray(Y,np.float64)
 if X.ndim!=2 or Y.ndim!=2 or len(X)!=len(Y)or not len(X)or not np.isfinite(X).all()or not np.isfinite(Y).all()or lam<0:raise ValueError('finite aligned numerical fit required')
 w=np.ones(len(X))/len(X)if weights is None else np.asarray(weights,np.float64)/np.sum(weights)
 if w.shape!=(len(X),)or not np.isfinite(w).all()or(w<0).any()or w.sum()<=0:raise ValueError('positive finite fit weights required')
 mu=w@X;ym=w@Y;A=(X-mu)*np.sqrt(w[:,None]);T=(Y-ym)*np.sqrt(w[:,None])
 scale=max(np.sum(A*A)/min(A.shape),1e-12)*lam
 if len(A)<A.shape[1]:W=A.T@np.linalg.solve(A@A.T+scale*np.eye(len(A)),T)
 else:W=np.linalg.solve(A.T@A+scale*np.eye(A.shape[1]),A.T@T)
 return W,ym-mu@W

def learn_maps(X9,X8,V,texts,y,inlist,tile9,tile8):
 V=V/np.linalg.norm(V,axis=1,keepdims=True);texts=texts/np.linalg.norm(texts,axis=1,keepdims=True)
 M,b=ridge(X9,V,.01);Wz=M@texts.T;bz=b@texts.T
 Z=X9@Wz+bz;pos=np.eye(len(texts),dtype=bool)[y];neg=inlist&~pos
 a=np.concatenate([Z[pos],Z[neg]]);t=np.concatenate([np.ones(pos.sum()),np.zeros(neg.sum())]);w=np.concatenate([np.full(pos.sum(),.5/pos.sum()),np.full(neg.sum(),.5/neg.sum())])
 am=w@a;tm=w@t;alpha=(w*(a-am)*(t-tm)).sum()/max((w*(a-am)**2).sum(),1e-12);beta=tm-alpha*am
 W=alpha*Wz;bias=alpha*bz+beta;prior=X9@W+bias
 counts=np.bincount(y,minlength=len(texts));weights=1/np.maximum(counts[y],1);weights/=weights.sum();target=np.eye(len(texts))[y]-prior
 X=np.concatenate([X9,tile9]);Y=np.concatenate([target,-(tile9@W+bias)]);ww=np.concatenate([weights,np.full(len(tile9),.5/max(len(tile9),1))])
 dW,db=ridge(X,Y,.1,ww);W9=W+dW;b9=bias+db
 X=np.concatenate([X8,tile8]);Y=np.concatenate([X9@W9+b9,np.zeros((len(tile8),len(texts)))]);W8,b8=ridge(X,Y,.01,ww)
 # +1 global view, -1 tiled views; affine cue used only as an extra soft gate.
 gx=np.concatenate([X9,tile9]);gy=np.concatenate([np.ones(len(X9)),-np.ones(len(tile9))])[:,None]
 gw=np.concatenate([np.full(len(X9),.5/len(X9)),np.full(len(tile9),.5/max(len(tile9),1))]);u,c=ridge(gx,gy,.01,gw)
 ux=np.concatenate([X8,tile8]);u8,c8=ridge(ux,gy,.01,gw)
 return dict(W9=W9,b9=b9,W8=W8,b8=b8,u=u[:,0],c=float(c[0]),u8=u8[:,0],c8=float(c8[0]),alpha=float(alpha),beta=float(beta))

def recover_gates(base,c39):
 W1,b1,W2,b2=c39
 def find(bias,column):
  hits=[i for i in range(len(b1))if abs(float(b1[i])-bias)<1e-3 and (column is None or np.max(np.abs(W2[:,i]-column))<1e-6)]
  if len(hits)!=1:raise ValueError('ambiguous source gate identity: '+str(bias))
  return hits[0]
 a=find(-100.,-b2);b=find(-101.,b2);c=find(-98.72,None);d=find(-99.72,None)
 if not np.array_equal(W1[a],W1[b])or not np.array_equal(W1[c],W1[d]):raise ValueError('source gate paired rows differ')
 r=W1[a].astype(np.float64)/200.;q=(200*r-W1[c])/115.
 changed=np.any(W2!=base[2],axis=0);slots=np.where(~changed)[0];slots=slots[np.argsort(np.sum(base[2][:,slots].astype(np.float64)**2,axis=0),kind='stable')]
 old=np.where(changed)[0];old=old[~np.isin(old,[a,b,c,d])]
 if len(old)!=116:raise ValueError('expected exactly58 previous/last hint pairs')
 return dict(r=r,q=q,slots=slots,old=old,special=[a,b,c,d])

def projector(c39,gates,maps,T8,T9,magnitude,route):
 W1,b1,W2,b2=[v.astype(np.float64).copy()for v in c39];n=len(T8)
 if route=='base':W2[:,gates['old']]=0.
 elif route!='c39':raise ValueError('unknown source route')
 if len(gates['slots'])<2*n:raise ValueError('insufficient editable projector units')
 for i in range(n):
  k=int(gates['slots'][i]);W1[k]=12*maps['W9'][:,i]+200*gates['r']+3*maps['u'];b1[k]=12*(maps['b9'][i]-.30)-200+3*(maps['c']-.93);W2[:,k]=magnitude*T9[i]
  k=int(gates['slots'][n+i]);W1[k]=12*maps['W8'][:,i]+200*gates['r']-115*gates['q']+3*maps['u8'];b1[k]=12*(maps['b8'][i]-.25)-200*.5486+3*(maps['c8']-.93);W2[:,k]=magnitude*T8[i]
 result=[x.astype(np.float32)for x in [W1,b1,W2,b2]]
 if any(not np.isfinite(v).all()for v in result):raise ValueError('nonfinite proposed tensor')
 return result

def letter_residual(H,head,y,lam=.1):
 logits=H@head.T/1.1;logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True);targets=np.eye(11)[y]-p
 A=H.T@H;A+=lam*max(np.trace(A)/H.shape[1],1e-12)*np.eye(H.shape[1]);return np.linalg.solve(A,H.T@targets).T
