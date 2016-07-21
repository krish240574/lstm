 z←lstm_main
 ⍝ http://arunmallya.github.io/writeups/nn/lstm/index.html#/2
 ⍝ size of input vector
 n←10
 ⍝ number of memory cells/hidden units
 d←4
 ⍝ Embed size for embedding layer
 esz←10
 ⍝ Word Embedding layer variables
 we_W←(d,esz)⍴((1000?1000)÷1000000)
 we_dW←(⍴we_W)⍴0
 we_x←(1,d)⍴0

 ⍝ Softmax layer vars
 sm_W←(d,d)⍴((1000?1000)÷1000000)
 sm_dW←(d,d)⍴0
 sm_pred←(1,d)⍴0

 ⍝ +1 for bias
 xt←((n),1)⍴((1000?1000)÷1000000)
 W←((4×d),(n+d))⍴((1000?1000)÷1000000)

 ⍝ per cell Ws and Us
 Wc←(n,d)⍴((1000?1000)÷1000000)
 Wi←(n,d)⍴((1000?1000)÷1000000)
 Wf←(n,d)⍴((1000?1000)÷1000000)
 Wo←(n,d)⍴((1000?1000)÷1000000)

 Uc←(d,d)⍴((1000?1000)÷1000000)
 Ui←(d,d)⍴((1000?1000)÷1000000)
 Uf←(d,d)⍴((1000?1000)÷1000000)
 Uo←(d,d)⍴((1000?1000)÷1000000)

 h0←(1,d)⍴((1000?1000)÷1000000)
 c0←(1,d)⍴((1000?1000)÷1000000)

 at←(1,d)⍴0
 athat←(1,d)⍴0
 it←(1,d)⍴0
 ft←(1,d)⍴0
 ot←(1,d)⍴0
 ct←(1,d)⍴0
 ht←(1,d)⍴0
 cprev←(1,d)⍴0
 hprev←(1,d)⍴0
 numInputUnits←d
 oloop←1
 sumW←0
 ⍝ toy example
 X←(1 2)⍴(2 1)
 eos←0
 Y←(1 1)⍴2
 ⍝ Forward pass
 ⍝ Input layers - LSTM+Emb - train with [EOS]+X - forward
 ⍝ reset output layers
 ⍝ Output layers - Emb+LSTM+Softmax - train with reversed([EOS]+Y)

 ⍝ Backward pass
 ⍝ Output layers - train with reversed(Y+[EOS])
 ⍝ Input layers - train with reversed(X)


 :While oloop≤10
     xt←((n),1)⍴((1000?1000)÷1000000)
     counter←1 ⍝ time counter
     :While counter≤numInputUnits
         ⍝ node t gets xt, t+1 gets xt+1...
         xt←eos,X[;counter]
         :If counter=1
             cprev←c0
             hprev←h0
         :Else
             cprev[;counter]←ct[;counter-1]
             hprev[;counter]←ht[;counter-1]
         :EndIf
         ⍝ LSTM - Forward pass
         athat[;counter]←+⌿((⍉Wc)+.×xt)+(Uc+.×⍉hprev)
         at[;counter]←7○athat[;counter]

         ithat←+⌿((⍉Wi)+.×xt)+(Ui+.×⍉hprev)
         it[;counter]←1÷(1+*(¯1×ithat))

         fthat←+⌿((⍉Wf)+.×xt)+(Uf+.×⍉hprev)
         ft[;counter]←1÷(1+*(¯1×fthat))

         othat←+⌿((⍉Wo)+.×xt)+(Uo+.×⍉hprev)
         ot[;counter]←1÷(1+*(¯1×othat))

         tmp←(it×at)+(ft×cprev)
         ct[;counter]←tmp[;counter]
         cprev[;counter]←ct[;counter]

         ht[;counter]←(ot[;counter])×(7○ct[;counter])
         hprev[;counter]←ht[;counter]

         ⍝ Word-embedding - forward pass
         we_x[;counter]←ht[;counter] ⍝ output of LSTM layer
         we_o←we_W[we_x[;counter]] ⍝ output of embedding layer

         ⍝ Softmax layer - forward pass
         sm_y←sm_W+.×we_o
         tmp←sm_y[⍋sm_y]
         sm_ymax←tmp[⍴sm_y]
         sm_y←*(sm_y-sm_ymax)
         sm_y←sm_y÷(+/sm_y)
         sm_pred[;counter]←sm_y
         sm_xt[;counter]←we_o

         counter←counter+1
     :EndWhile
      ⍝ reverse Y+[EOS]
     delta←⌽(Y,eos)
 ⍝ backward pass - softmax
     :While counter≥1
         :If counter>1
             sm_d←sm_pred[;counter-1]
         :Else
             sm_d←sm_pred[;counter]
         :EndIf
         sm_d[target]←sm_d[target]-1
         :If counter>1
             sm_dW←sm_d×.∘sm_xt[;counter-1] ⍝ outer product
         :Else
             sm_dW←sm_d×.∘sm_xt[;counter]
         :EndIf
         sm_delta←sm_W+.×sm_d
         counter←counter-1
     :EndWhile
     dExdH←sm_delta ⍝ error derivative
 ⍝   backward pass  - LSTM
     dExdct←(1,d)⍴0
     dExdot←(1,d)⍴0
     dExdit←(1,d)⍴0
     dExdft←(1,d)⍴0
     dExdat←(1,d)⍴0
     dExdct←(1,d)⍴0
     dzt←(1,d)⍴0
     I←(1,d)⍴0
     dExdWt←(1,d)⍴0
     dExdH←((⍴ht))⍴((1000?1000)÷1000000) ⍝ random numbers for err derivative, for now

     counter←numInputUnits
     :While counter≥1
         dExdot[;counter]←dExdH[;counter]×(7○ct[;counter])

         dExdct[;counter]←dExdct[;counter]+dExdH[;counter]×ot[;counter]×(1-(7○ct[;counter])*2)

         dExdit[;counter]←dExdct[;counter]×at[;counter]
         :If counter>1
             dExdft[;counter]←dExdct[;counter]×ct[;counter-1]
         :Else
             dExdft[;counter]←dExdct[;counter]×c0[;counter]
         :EndIf
         dExdat[;counter]←dExdct[;counter]×it[;counter]
         dExdcprev←dExdct[;counter]×ft[;counter]

         dExdahat←dExdat[;counter]×(1-(7○athat[;counter])*2)
         dExdihat←dExdit[;counter]×it[;counter]×(1-it[;counter])
         dExdfhat←dExdft[;counter]×ft[;counter]×(1-ft[;counter])
         dExdohat←dExdot[;counter]×ot[;counter]×(1-ot[;counter])
         dzt[;counter]←⊂(1,d)⍴(dExdahat dExdihat dExdfhat dExdohat)

         :If counter>1
             I[;counter]←⊂(2 1)⍴((xt)(ht[;counter-1]))
         :Else
             I[;counter]←⊂(2 1)⍴((xt)(h0[;counter]))
         :EndIf
         dExdWt[;counter]←⊂(⍉↑dzt[;counter])+.×⍉↑I[;counter]
         counter←counter-1
     :EndWhile
     sumW←sumW+dExdWt



     oloop←oloop+1
 :EndWhile
