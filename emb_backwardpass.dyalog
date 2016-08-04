 z←t emb_backwardpass delta
 :If t>1
     x←we_x[LAYERNUM;;t-1]
 :Else
     x←we_x[LAYERNUM;;t]
 :EndIf

 we_dW[LAYERNUM;x;]←((1,d)⍴,⊃we_dW[LAYERNUM;x;])+delta

 z←0
