 z←counter lstm_backwardpass xt
 dExdot[LAYERNUM;;counter]←dExdH[LAYERNUM;;counter]×(7○ct[LAYERNUM;;counter])

 dExdct[LAYERNUM;;counter]←dExdct[LAYERNUM;;counter]+dExdH[LAYERNUM;;counter]×ot[LAYERNUM;;counter]×(1-(7○ct[LAYERNUM;;counter])*2)

 dExdit[LAYERNUM;;counter]←dExdct[LAYERNUM;;counter]×at[LAYERNUM;;counter]
 :If counter>1
     dExdft[LAYERNUM;;counter]←dExdct[LAYERNUM;;counter]×ct[LAYERNUM;;counter-1]
 :Else
     dExdft[LAYERNUM;;counter]←dExdct[LAYERNUM;;counter]×c0[LAYERNUM;;counter]
 :EndIf
 dExdat[LAYERNUM;;counter]←dExdct[LAYERNUM;;counter]×it[LAYERNUM;;counter]
 dExdcprev←dExdct[LAYERNUM;;counter]×ft[LAYERNUM;;counter]

 dExdahat←dExdat[LAYERNUM;;counter]×(1-(7○athat[LAYERNUM;;counter])*2)
 dExdihat←dExdit[LAYERNUM;;counter]×it[LAYERNUM;;counter]×(1-it[LAYERNUM;;counter])
 dExdfhat←dExdft[LAYERNUM;;counter]×ft[LAYERNUM;;counter]×(1-ft[LAYERNUM;;counter])
 dExdohat←dExdot[LAYERNUM;;counter]×ot[LAYERNUM;;counter]×(1-ot[LAYERNUM;;counter])
 dzt[;counter]←⊂(1,d)⍴(dExdahat dExdihat dExdfhat dExdohat)

 :If counter>1
     I[LAYERNUM;;counter]←⊂(2 1)⍴((input)(ht[LAYERNUM;;counter-1]))
 :Else
     I[LAYERNUM;;counter]←⊂(2 1)⍴((input)(h0[LAYERNUM;;counter]))
 :EndIf
 dExdWt[LAYERNUM;;counter]←⊂(⍉↑dzt[LAYERNUM;;counter])+.×⍉↑I[LAYERNUM;;counter]
