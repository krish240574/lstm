 z←lstm_main
 ⍝ http://arunmallya.github.io/writeups/nn/lstm/index.html#/2
 ⍝ size of input vector

 n←10
 ⍝ number of memory cells/hidden units
 d←10

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
 ⍝ Forward pass


 :While oloop≤10
     counter←1 ⍝ time counter
     xt←((n),1)⍴((1000?1000)÷1000000)
     :While counter≤numInputUnits
         ⍝ node t gets xt, t+1 gets xt+1...

         :If counter=1
             cprev←c0
             hprev←h0
         :Else
             cprev[;counter]←ct[;counter-1]
             hprev[;counter]←ht[;counter-1]
         :EndIf
         x←(1 1)⍴xt[counter;]
         ⍝ LSTM - Forward pass
         athat[;counter]←+⌿((⍉Wc)+.×x)+(Uc+.×⍉hprev)
         at[;counter]←7○athat[;counter]

         ithat←+⌿((⍉Wi)+.×x)+(Ui+.×⍉hprev)
         it[;counter]←1÷(1+*(¯1×ithat))

         fthat←+⌿((⍉Wf)+.×x)+(Uf+.×⍉hprev)
         ft[;counter]←1÷(1+*(¯1×fthat))

         othat←+⌿((⍉Wo)+.×x)+(Uo+.×⍉hprev)
         ot[;counter]←1÷(1+*(¯1×othat))

         tmp←(it×at)+(ft×cprev)
         ct[;counter]←tmp[;counter]
         cprev[;counter]←ct[;counter]

         ht[;counter]←(ot[;counter])×(7○ct[;counter])
         hprev[;counter]←ht[;counter]

         counter←counter+1
     :EndWhile

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
             I[;counter]←⊂(2 1)⍴((xt[counter;])(ht[;counter-1]))
         :Else
             I[;counter]←⊂(2 1)⍴((xt[counter;])(h0[;counter]))
         :EndIf
         dExdWt[;counter]←⊂(⍉↑dzt[;counter])+.×⍉↑I[;counter]
         counter←counter-1
     :EndWhile

     sumW←sumW+dExdWt
     oloop←oloop+1
 :EndWhile
