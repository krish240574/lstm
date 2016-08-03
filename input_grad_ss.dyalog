 z←input_grad_ss
 ⍝ all input layers
 ⍝ lstm, embedding layers

 ⍝ LSTM layer
 gradsum←+⌿((deltaWa[LAYERNUM;;]*2)+(deltaWf[LAYERNUM;;]*2)+(deltaWi[LAYERNUM;;]*2)+(deltaWo[LAYERNUM;;]*2))
 gradsum←+/(gradsum+(+⌿(deltaUa[LAYERNUM;;]*2)+(deltaUf[LAYERNUM;;]*2)+(deltaUi[LAYERNUM;;]*2)+(deltaUo[LAYERNUM;;]*2)))

 ⍝ Embedding layer
 gradsum←gradsum++⌿(we_dW[LAYERNUM;;x]*2)
