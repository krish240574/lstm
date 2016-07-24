 z←counter lstm_forwardpass x

 :If counter=1
     cprev[LAYERNUM;;]←c0[LAYERNUM;;]
     hprev[LAYERNUM;;]←h0[LAYERNUM;;]
 :Else
     cprev[LAYERNUM;;counter]←ct[LAYERNUM;;counter-1]
     hprev[LAYERNUM;;counter]←ht[LAYERNUM;;counter-1]
 :EndIf
 athat[LAYERNUM;;counter]←+⌿(Wc[LAYERNUM;;]+.×⍉x)+(Uc[LAYERNUM;;]+.×⍉hprev[LAYERNUM;;])
 at[LAYERNUM;;counter]←7○athat[LAYERNUM;;counter]

 ithat←+⌿(Wi[LAYERNUM;;]+.×⍉x)+(Ui[LAYERNUM;;]+.×⍉hprev[LAYERNUM;;])
 it[LAYERNUM;;counter]←1÷(1+*(¯1×ithat))

 fthat←+⌿(Wf[LAYERNUM;;]+.×⍉x)+(Uf[LAYERNUM;;]+.×⍉hprev[LAYERNUM;;])
 ft[LAYERNUM;;counter]←1÷(1+*(¯1×fthat))

 othat←+⌿(Wo[LAYERNUM;;]+.×⍉x)+(Uo[LAYERNUM;;]+.×⍉hprev[LAYERNUM;;])
 ot[LAYERNUM;;counter]←1÷(1+*(¯1×othat))

 tmp←(it[LAYERNUM;;]×at[LAYERNUM;;])+(ft[LAYERNUM;;]×cprev[LAYERNUM;;])
 ct[LAYERNUM;;]←tmp
 cprev[LAYERNUM;;counter]←ct[LAYERNUM;;counter]

 ht[LAYERNUM;;counter]←(ot[LAYERNUM;;counter])×(7○ct[LAYERNUM;;counter])
 hprev[LAYERNUM;;counter]←ht[LAYERNUM;;counter]

 z←ht
