 z←counter emb_backwardpass delta
 :If counter>1
     x←we_x[LAYERNUM;;counter-1]
 :Else
     x←we_x←[LAYERNUM;;counter]
 :EndIf

 we_dW[LAYERNUM;;x]←delta
