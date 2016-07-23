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

 ⍝ toy example
 X←(1 2)⍴(2 1)
 eos←0
 Y←(1 1)⍴2
 ⍝ Forward pass
 ⍝ Input layers - LSTM+Emb - train with [EOS]+X - forward
 ⍝ reset output layers
 ⍝ Output layers - Emb+LSTM+Softmax - train with reversed([EOS]+Y)


 counter←1
 :While counter≤d
     xt←(1 2)⍴(eos,X[;counter])
     :If counter=1
         cprev[INPUT;;]←c0[INPUT;;]
         hprev[INPUT;;]←h0[INPUT;;]
     :Else
         cprev[INPUT;;counter]←ct[INPUT;;counter-1]
         hprev[INPUT;;counter]←ht[INPUT;;counter-1]
     :EndIf
     athat[INPUT;;counter]←+⌿(Wc[INPUT;;]+.×⍉xt)+(Uc[INPUT;;]+.×⍉hprev[INPUT;;])
     at[INPUT;;counter]←7○athat[INPUT;;counter]

     ithat←+⌿(Wi[INPUT;;]+.×⍉xt)+(Ui[INPUT;;]+.×⍉hprev[INPUT;;])
     it[INPUT;;counter]←1÷(1+*(¯1×ithat))

     fthat←+⌿(Wf[INPUT;;]+.×⍉xt)+(Uf[INPUT;;]+.×⍉hprev[INPUT;;])
     ft[INPUT;;counter]←1÷(1+*(¯1×fthat))

     othat←+⌿(Wo[INPUT;;]+.×⍉xt)+(Uo[INPUT;;]+.×⍉hprev[INPUT;;])
     ot[INPUT;;counter]←1÷(1+*(¯1×othat))

     tmp←(it[INPUT;;]×at[INPUT;;])+(ft[INPUT;;]×cprev[INPUT;;])
     ct[INPUT;;]←tmp
     cprev[INPUT;;counter]←ct[INPUT;;counter]

     ht[INPUT;;counter]←(ot[INPUT;;counter])×(7○ct[INPUT;;counter])
     hprev[INPUT;;counter]←ht[INPUT;;counter]

     counter←counter+1
 :EndWhile
