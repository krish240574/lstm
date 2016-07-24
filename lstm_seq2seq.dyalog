 z←lstm_seq2seq
 n←2 ⍝ size of input sequence
 d←2  ⍝ number of input units/memory cells
 esz←10 ⍝ size of embedding layer

 INPUT←1
 OUTPUT←2
⍝ Word embedding layer
 we_W←(2,d,esz)⍴((1000?1000)÷1000000) ⍝ we need 2 layers
 we_dW←(⍴we_W)⍴0
 we_x←(2,d)⍴0

⍝ Softmax layer
 sm_W←(d,d)⍴((1000?1000)÷1000000)
 sm_dW←(d,d)⍴0
 sm_pred←(1,d)⍴0


⍝ LSTM layers
 W←(2,(4×d),(n+d))⍴((1000?1000)÷1000000)

 ⍝ per cell Ws and Us
 Wc←(2,n,d)⍴((1000?1000)÷1000000)
 Wi←(2,n,d)⍴((1000?1000)÷1000000)
 Wf←(2,n,d)⍴((1000?1000)÷1000000)
 Wo←(2,n,d)⍴((1000?1000)÷1000000)
 Uc←(2,d,d)⍴((1000?1000)÷1000000)
 Ui←(2,d,d)⍴((1000?1000)÷1000000)
 Uf←(2,d,d)⍴((1000?1000)÷1000000)
 Uo←(2,d,d)⍴((1000?1000)÷1000000)

 h0←(2,1,d)⍴((1000?1000)÷1000000)
 c0←(2,1,d)⍴((1000?1000)÷1000000)

 at←(2,1,d)⍴0
 athat←(2,1,d)⍴0
 it←(2,1,d)⍴0
 ft←(2,1,d)⍴0
 ot←(2,1,d)⍴0
 ct←(2,1,d)⍴0
 ht←(2,1,d)⍴0
 cprev←(2,1,d)⍴0
 hprev←(2,1,d)⍴0

 ⍝   backward pass  - LSTM
 dExdct←(2,1,d)⍴0
 dExdot←(2,1,d)⍴0
 dExdit←(2,1,d)⍴0
 dExdft←(2,1,d)⍴0
 dExdat←(2,1,d)⍴0
 dExdct←(2,1,d)⍴0
 dzt←(2,1,d)⍴0
 I←(2,1,d)⍴0
 dExdWt←(1,d)⍴0

 ⍝ toy example
 XX←(1 2)⍴(2 1)
 eos←0
 YY←(1 1)⍴2
 ⍝ Forward pass
 ⍝ Input layers - LSTM+Emb - train with [EOS]+X - forward
 ⍝ reset output layers
 ⍝ Output layers - Emb+LSTM+Softmax - train with reversed([EOS]+Y)

 LAYERNUM←INPUT
 ⍝ input layers - forward pass
 counter←1
 :While counter≤d
     h←counter input_forward X[;counter]
     counter←counter+1
 :EndWhile

 LAYERNUM←OUTPUT
 ⍝ output layers - forward pass
 Y←eos,YY
 counter←1
 :While counter≤d
     h←Y[;counter]
     h←counter output_forward h
     counter←counter+1
 :EndWhile


 ⍝ output layers - backward pass
 Y←⌽(YY,eos)
 counter←d
 :While counter≥1
     delta←Y[;counter]
     delta←counter output_backward delta
     counter←counter-1
 :EndWhile

 LAYERNUM←INPUT
 X←⌽(XX)
 counter←d
 :While counter≥1
     delta←(1,esz)⍴0
     delta←counter input_backward delta
     counter←counter-1
 :EndWhile

 ⍝ gradient clipping
 ⍝ LAYERNUM←INPUT
 ⍝ dExdH←((⍴ht))⍴((1000?1000)÷1000000) ⍝ random numbers for err derivative, for now

 ⍝ counter←d
