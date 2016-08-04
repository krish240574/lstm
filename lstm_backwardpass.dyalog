 z←t lstm_backwardpass delta
 dExdot[LAYERNUM;;]←(⍉delta)×(7○ct[LAYERNUM;;])
 ⍝dExdot[LAYERNUM;;t]←dExdH[LAYERNUM;;t]×(7○ct[LAYERNUM;;t])

 dExdct[LAYERNUM;;]←dExdct[LAYERNUM;;]+(⍉delta)×ot[LAYERNUM;;]×(1-(7○ct[LAYERNUM;;])*2)

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
 ⍝ layer cell, one needs to shift values of the inputs, h
 ⍝ with values learnt from bptt in this time step

 ⍝ 1. Calculate the increments for Wa,f,i,o (outer product, each derivative with delta
 ⍝ 2. Update the weights
 deltaWa[LAYERNUM;;]←delta∘.×dExdahat
 Wc[LAYERNUM;t;]←((1,d)⍴Wc[LAYERNUM;;t])+(1,d)⍴,⊃deltaWa[LAYERNUM;;]
 deltaWi[LAYERNUM;;]←delta∘.×dExdihat
 Wi[LAYERNUM;t;]←((1,d)⍴Wi[LAYERNUM;;t])+(1,d)⍴,⊃deltaWi[LAYERNUM;;]
 deltaWf[LAYERNUM;;]←delta∘.×dExdfhat
 Wf[LAYERNUM;t;]←((1,d)⍴Wf[LAYERNUM;;t])+(1,d)⍴,⊃deltaWf[LAYERNUM;;]
 deltaWo[LAYERNUM;;]←delta∘.×dExdohat
 Wo[LAYERNUM;t;]←((1,d)⍴Wo[LAYERNUM;;t])+(1,d)⍴,⊃deltaWo[LAYERNUM;;]

 :If t>1
     tmp←ct[LAYERNUM;;t-1]
 :Else
     tmp←c0[LAYERNUM;;1]  ⍝ or t since t will be 1 here
 :EndIf
 ⍝ to correct dimensions here
 deltaUa[LAYERNUM;;t]←tmp∘.×⍉dExdahat
 Ua←Ua+deltaUa
 deltaUi←tmp∘.×⍉dExdihat
 Ui←Ui+deltaUi
 deltaUf←tmp∘.×⍉dExdfhat
 Uf←Uf+deltaUf
 deltaUo←tmp∘.×⍉dExdohat
 Uo←Uo+deltaUo

 ⍝ Now to do the same for hprev
 hprevtoa[LAYERNUM;;]←Ua[LAYERNUM;;]+.×⍉dExdahat
 hprevtoi[LAYERNUM;;]←Ui[LAYERNUM;;]+.×⍉dExdihat
 hprevtof[LAYERNUM;;]←Uf[LAYERNUM;;]+.×⍉dExdfhat
 hprevtoo[LAYERNUM;;]←Uo[LAYERNUM;;]+.×⍉dExdohat
 dhprev[LAYERNUM;;]←hprevtoa[LAYERNUM;;]+hprevtof[LAYERNUM;;]+hprevtoi[LAYERNUM;;]+hprevtoo[LAYERNUM;;]
 hprev[LAYERNUM;;]←hprev[LAYERNUM;;]+(1,d)⍴dhprev

 ⍝ Now to return delta h
 dx←Wc[LAYERNUM;;t]+.×dExdahat
 dx←dx+Wf[LAYERNUM;;t]+.×dExdfhat
 dx←dx+Wi[LAYERNUM;;t]+.×dExdihat
 dx←dx+Wo[LAYERNUM;;t]+.×dExdohat

 z←dx
