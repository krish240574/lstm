 z←t lstm_forwardpass x

 :If t=1
     cprev[LAYERNUM;;]←c0[LAYERNUM;;]
     hprev[LAYERNUM;;]←h0[LAYERNUM;;]
 :Else
     cprev[LAYERNUM;;t]←ct[LAYERNUM;;t-1]
     hprev[LAYERNUM;t;]←ht[LAYERNUM;t-1;]
 :EndIf
 athat[LAYERNUM;;t]←+/((1,d)⍴(Wc[LAYERNUM;;]+.×x))+(1,d)⍴(Uc[LAYERNUM;;]+.×⍉hprev[LAYERNUM;t;])
 at[LAYERNUM;;t]←7○athat[LAYERNUM;;t]

 ithat←+/((1,d)⍴(Wi[LAYERNUM;;]+.×x))+(1,d)⍴(Ui[LAYERNUM;;]+.×⍉hprev[LAYERNUM;;])
 it[LAYERNUM;;t]←1÷(1+*(¯1×ithat))

 fthat←+/((1,d)⍴(Wf[LAYERNUM;;]+.×x))+(1,d)⍴(Uf[LAYERNUM;;]+.×⍉hprev[LAYERNUM;;])
 ft[LAYERNUM;;t]←1÷(1+*(¯1×fthat))

 othat←+/((1,d)⍴(Wo[LAYERNUM;;]+.×x))+(1,d)⍴(Uo[LAYERNUM;;]+.×⍉hprev[LAYERNUM;;])
 ot[LAYERNUM;;t]←1÷(1+*(¯1×othat))

 tmp←(it[LAYERNUM;;]×at[LAYERNUM;;])+(ft[LAYERNUM;;]×cprev[LAYERNUM;;])
 ct[LAYERNUM;;]←tmp
 cprev[LAYERNUM;;t]←ct[LAYERNUM;;t]

 ht[LAYERNUM;;t]←(ot[LAYERNUM;;t])×(7○ct[LAYERNUM;;t])
 hprev[LAYERNUM;;t]←ht[LAYERNUM;;t]

 z←ht[LAYERNUM;;t] ⍝ return output of 1 time step
