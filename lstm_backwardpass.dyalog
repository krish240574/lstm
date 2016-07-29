 z←t lstm_backwardpass xt
 dExdot[LAYERNUM;;t]←dExdH[LAYERNUM;;t]×(7○ct[LAYERNUM;;t])

 dExdct[LAYERNUM;;t]←dExdct[LAYERNUM;;t]+dExdH[LAYERNUM;;t]×ot[LAYERNUM;;t]×(1-(7○ct[LAYERNUM;;t])*2)

 dExdit[LAYERNUM;;t]←dExdct[LAYERNUM;;t]×at[LAYERNUM;;t]
 :If t>1
     dExdft[LAYERNUM;;t]←dExdct[LAYERNUM;;t]×ct[LAYERNUM;;t-1]
 :Else
     dExdft[LAYERNUM;;t]←dExdct[LAYERNUM;;t]×c0[LAYERNUM;;t]
 :EndIf
 dExdat[LAYERNUM;;t]←dExdct[LAYERNUM;;t]×it[LAYERNUM;;t]
 dExdcprev←dExdct[LAYERNUM;;t]×ft[LAYERNUM;;t]

 dExdahat←dExdat[LAYERNUM;;t]×(1-(7○athat[LAYERNUM;;t])*2)
 dExdihat←dExdit[LAYERNUM;;t]×it[LAYERNUM;;t]×(1-it[LAYERNUM;;t])
 dExdfhat←dExdft[LAYERNUM;;t]×ft[LAYERNUM;;t]×(1-ft[LAYERNUM;;t])
 dExdohat←dExdot[LAYERNUM;;t]×ot[LAYERNUM;;t]×(1-ot[LAYERNUM;;t])

 ⍝ Since the output of this cell(bproped) is going as input to the embedding
 ⍝ layer cell, one needs to shift values of the inputs, xt
 ⍝ with values learnt from bptt in this time step

 ⍝ 1. Calculate the increments for Wa,f,i,o (outer product, each derivative with xt
 ⍝ 2. Update the weights
 deltaWa←xt∘.×dExdahat
 Wa[LAYERNUM;;t]←Wa[LAYERNUM;;t]+deltaWa
 deltaWi←xt∘.×dExdihat
 Wi[LAYERNUM;;t]←Wi[LAYERNUM;;t]+deltaWi
 deltaWf←xt∘.×dExdfhat
 Wf[LAYERNUM;;t]←Wf[LAYERNUM;;t]+deltaWf
 deltaWo←xt∘.×dExdohat
 Wo[LAYERNUM;;t]←Wo[LAYERNUM;;t]+deltaWo

 :If t>1
     tmp←ct[LAYERNUM;;t-1]
 :Else
     tmp←c0[LAYERNUM;;1]  ⍝ or t since t will be 1 here
 :EndIf
 ⍝ to correct dimensions here
 deltaUa←+⌿(d,d,d)⍴,tmp∘.×⍉dExdahat
 Ua←Ua+deltaUa
 deltaUi←+⌿(d,d,d)⍴tmp∘.×⍉dExdihat
 Ua←Ui+deltaUi
 deltaUf←+⌿(d,d,d)⍴tmp∘.×⍉dExdfhat
 Uf←Uf+deltaUf
 deltaUo←+⌿(d,d,d)⍴tmp∘.×⍉dExdohat
 Uo←Uo+deltaUo

 ⍝ Now to do the same for hprev - to correct dimensions here too
 hprevtoa[LAYERNUM;;t]←+⌿(d,d,d)⍴Ua[LAYERNUM;;t]∘.×⍉dExdahat
 hprevtoi[LAYERNUM;;t]←+⌿(d,d,d)⍴Ui[LAYERNUM;;t]∘.×⍉dExdihat
 hprevtof[LAYERNUM;;t]←+/(d,d,d)⍴Uf[LAYERNUM;;t]∘.×⍉dExdfhat
 hprevtoo[LAYERNUM;;t]←+⌿(d,d,d)⍴Uo[LAYERNUM;;t]∘.×⍉dExdohat
 dhprev←hprevtoa[LAYERNUM;;t]+hprevtof[LAYERNUM;;t]+hprevtoi[LAYERNUM;;t]+hprevtoo[LAYERNUM;;t]
 hprev[LAYERNUM;;t]←hprev[LAYERNUM;;t]+dhprev[LAYERNUM;;t]

 ⍝ Now to return delta xt
 dx←Wa[LAYERNUM;;t]+.×dExdahat
 dx←dx+Wf[LAYERNUM;;t]+.×dExdfhat
 dx←dx+Wi[LAYERNUM;;t]+.×dExdihat
 dx←dx+Wo[LAYERNUM;;t]+.×dExdohat

 z←dx
