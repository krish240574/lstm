 z←lstm_seq2seq

 d←10   ⍝ number of hidden units/memory cells
 esz←10 ⍝ size of embedding layer
 n←2    ⍝ size of input sequence
 v←4    ⍝ vocab size

 INPUT←1
 OUTPUT←2
 ⍝ Word embedding layer
 we_W←(2,v,esz)⍴((1000?1000)÷1000000) ⍝ we need 2 layers
 we_dW←(⍴we_W)⍴0
 we_x←'null'

 ⍝ Softmax layer
 sm_W←(n,d)⍴((1000?1000)÷1000000)
 sm_dW←(n,d)⍴0
 sm_pred←'null'
 sm_xt←'null'
 sm_target←'null'


 ⍝ LSTM layers
 W←(2,(4×d),(n+d))⍴((1000?1000)÷1000000)

 ⍝ per cell Ws and Us
 Wa←(2,esz,d)⍴((1000?1000)÷1000000)
 Wi←(2,esz,d)⍴((1000?1000)÷1000000)
 Wf←(2,esz,d)⍴((1000?1000)÷1000000)
 Wo←(2,esz,d)⍴((1000?1000)÷1000000)
 Ua←(2,d,d)⍴((1000?1000)÷1000000)
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
 hprevtoa←(2,1,d)⍴0
 hprevtoi←(2,1,d)⍴0
 hprevtof←(2,1,d)⍴0
 hprevtoo←(2,1,d)⍴0
 dhprev←(2,1,d)⍴0

 deltaWa←(2,d,d)⍴0
 deltaWf←(2,d,d)⍴0
 deltaWi←(2,d,d)⍴0
 deltaWo←(2,d,d)⍴0
 deltaUa←(2,d,d)⍴0
 deltaUf←(2,d,d)⍴0
 deltaUi←(2,d,d)⍴0
 deltaUo←(2,d,d)⍴0


 ⍝   backWcrd pass  - LSTM
 dExdct←(2,1,d)⍴0
 dExdot←(2,1,d)⍴0
 dExdit←(2,1,d)⍴0
 dExdft←(2,1,d)⍴0
 dExdat←(2,1,d)⍴0
 dzt←(2,1,d)⍴0
 I←(2,1,d)⍴0
 dExdWt←(1,d)⍴0

 ⍝ toy example
 XX←(1 2)⍴(2 1)
 eos←0
 YY←(1 1)⍴2
 ⍝ ForWcrd pass
 ⍝ Input layers - LSTM+Emb - train with [EOS]+X - forWcrd
 ⍝ reset output layers
 ⍝ Output layers - Emb+LSTM+Softmax - train with reversed([EOS]+Y)

 LAYERNUM←INPUT
 ⍝ input layers - forward pass
 t←1
 ;
 :While t≤n
     h←t input_forward XX[;t]
     t←t+1
 :EndWhile

 LAYERNUM←OUTPUT
 ⍝ output layers - forward pass
 Y←eos,YY
 t←1
 :While t≤n
     h←Y[;t]
     h←t output_forward h
     t←t+1
 :EndWhile


 ⍝ output layers - backward pass
 Y←⌽(YY,eos)
 t←n
 :While t≥1
     delta←Y[;t]
     delta←t output_backward delta
     t←t-1
 :EndWhile
 ;
 LAYERNUM←INPUT
 X←⌽(XX)
 t←(¯1↑⍴X)
 :While t≥1
     delta←(1,esz)⍴0
     delta←t input_backward delta
     t←t-1
 :EndWhile

 ⍝ gradient clipping
 clip_grad←5 ⍝ Very high value, if norm is greater than this, then clip
 ⍝ square and sum all gradients of all layers
 LAYERNUM←INPUT
 grad_ss←input_grad_ss
 LAYERNUM←OUTPUT
 grad_ss←grad_ss+output_grad_ss
 grad_norm←(grad_ss)*0.5

 :If gradsum>clip_grad
     (Wc Wf Wi Wo)←(Wc Wf Wi Wo we_W sm_W)÷(gradsum÷clip_grad)
     (Ua Uf Ui Uo)←(Ua Uf Ui Uo)÷(gradsum÷clip_grad)
 :EndIf
 (Wc Wf Wi Wo we_W sm_W)←(Wc Wf Wi Wo we_W sm_W)-lr×(Wc Wf Wi Wo we_W sm_W)
 (Ua Uf Ui Uo)←(Ua Uf Ui Uo)-lr×(Ua Uf Ui Uo)
