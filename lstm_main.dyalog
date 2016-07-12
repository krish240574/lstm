 z←lstm_main
 ⍝ http://arunmallya.github.io/writeups/nn/lstm/index.html#/2
 ⍝ size of input vector
 n←10
 ⍝ number of memory cells/hidden units
 d←4
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
 it←(1,d)⍴0
 ft←(1,d)⍴0
 ot←(1,d)⍴0
 ct←(1,d)⍴0
 ht←(1,d)⍴0
 cprev←(1,d)⍴0
 hprev←(1,d)⍴0
 numInputUnits←d
 counter←1
 :While counter≤numInputUnits
     :If counter=1
         cprev←c0
         hprev←h0
     :Else
         cprev[;counter]←ct[;counter-1]
         hprev[;counter]←ht[;counter-1]
     :EndIf
 ⍝ Forward pass
     athat←+⌿((⍉Wc)+.×xt)+(Uc+.×⍉hprev)
     at[;counter]←7○athat

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

     counter←counter+1
 :EndWhile



 ⍝backward pass
 dExdct←(1,d)⍴0
 dExdot←(1,d)⍴0
 dExdit←(1,d)⍴0
 dExdft←(1,d)⍴0
 dExdat←(1,d)⍴0
 dExdct←(1,d)⍴0
 dzt←(1,d)⍴0
 dExdH←((⍴ht))⍴((1000?1000)÷1000000) ⍝ random numbers for err derivative, for now

 counter←numInputUnits
 :While counter≥1
     dExdot[;counter]←dExdH×(7○ct)

     dExdct[;counter]←dExdct[;counter]+dExdH×ot[;counter]×(1-(7○ct[;counter])*2)

     dExdit[;counter]←dExdct[;counter]×at[;counter]
     :If counter>1
         dExdft[;counter]←dExdct[;counter]×ct[;counter-1]
     :Else
         dExdft[;counter]←dExdct[;counter]×c0
     :EndIf
     dExdat[;counter]←dExdct[;counter]×it[;counter]
     dExdcprev←dExdct[;counter]×ft[;counter]

     dExdahat←dExdat[;counter]×(1-(7○athat[;counter])*2)
     dExdihat←dExdit[;counter]×it[;counter]×(1-it[;counter])
     dExdfhat←dExdft[;counter]×ft[;counter]×(1-dExdft[;counter])
     dExdohat←dExdot[;counter]×ot[;counter]×(1-dExdot[;counter])
     dzt[;counter]←⍉(dExdahat,dExdihat,dExdfhat,dExdohat)

     I[;counter]←(2 1)⍴((xt)(ht[;counter-1]))
     dExdWt[;counter]←dzt[;counter]+.×⍉(I[;counter])

     counter←counter-1
 :EndWhile
