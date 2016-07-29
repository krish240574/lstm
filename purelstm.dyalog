 z←lstm_main
 ⍝ size of input vector
 n←10
 ⍝ number of memory cells/hidden units
 d←3

 ⍝ +1 for bias
 xt←((n),1)⍴((1000?1000)÷1000000)
 W←((4×d),(n+d))⍴((1000?1000)÷1000000)

 ⍝ per cell Ws and Us
 Wa←(n,d)⍴((1000?1000)÷1000000) ⍝ weights from input to at
 Wi←(n,d)⍴((1000?1000)÷1000000) ⍝ weights from input to it
 Wf←(n,d)⍴((1000?1000)÷1000000) ⍝ weights from input to ft
 Wo←(n,d)⍴((1000?1000)÷1000000) ⍝ weights from input to ot

 Ua←(d,d)⍴((1000?1000)÷1000000) ⍝ weights from ht-1 to at
 Ui←(d,d)⍴((1000?1000)÷1000000) ⍝ weights from ht-1 to it
 Uf←(d,d)⍴((1000?1000)÷1000000) ⍝ weights from ht-1 to ft
 Uo←(d,d)⍴((1000?1000)÷1000000) ⍝ weights from ht-1 to ot

 h0←(d,d)⍴((1000?1000)÷1000000)
 c0←(1,d)⍴((1000?1000)÷1000000)

 at←(1,d)⍴0 ⍝ 1 at per node
 athat←(1,d)⍴0
 it←(1,d)⍴0 ⍝ 1 it per node
 ft←(1,d)⍴0
 ot←(1,d)⍴0 ⍝ 1 it per node
 ct←(1,d)⍴0 ⍝ 1 it per node
 ht←(d,d)⍴0
 cprev←(d,d)⍴0
 hprev←(d,d)⍴0
 numInputUnits←d
 oloop←1
 sumW←0

 ⍝ Forward pass
 ⍝ :While oloop≤10
 t←1 ⍝ time t
 xt←((n),1)⍴((1000?1000)÷1000000)
 :While t≤numInputUnits
      ⍝ node t gets xt, t+1 gets xt+1...
     x←(1 1)⍴xt[t;]
     :If t=1
         cprev←c0
         hprev[t;]←h0[t;]
     :Else
         cprev[;t]←ct[;t-1]
         hprev[t;]←ht[t-1;]
         ⍝cprev[;t]←ct[;t-1]
         ⍝hprev[;t]←ht[;t-1]
     :EndIf

     athat[;t]←(+⌿(Wa+.×x))+(+⌿(Ua+.×⍉hprev[t;]))
     at[;t]←7○athat[;t]

     ithat←(+⌿(Wi+.×x))+(+⌿(Ui+.×⍉hprev[t;]))
     it[;t]←1÷(1+*(¯1×ithat))

     fthat←(+⌿(Wf+.×x))+(+⌿(Uf+.×⍉hprev[t;]))
     ft[;t]←1÷(1+*(¯1×fthat))

     othat←(+⌿(Wo+.×x))+(+/(Uo+.×⍉hprev[t;]))
     ot[;t]←1÷(1+*(¯1×othat))

     tmp←(it×at)+(ft×cprev[;t])
     ct[;t]←tmp[;t]
     ⍝ct[;t]←(1 3)⍴tmp

     ⍝ht[t;]←(ot[;t])×(7○ct[;t])
     ht[t;]←ot×7○ct[;t]

     t←t+1
 :EndWhile

 ⍝   backward pass  - LSTM
 dExdct←(1,d)⍴0
 dExdot←(1,d)⍴0
 dExdit←(1,d)⍴0
 dExdft←(1,d)⍴0
 dExdat←(1,d)⍴0
 dExdct←(1,d)⍴0
 dExdahat←(1,d)⍴0
 dExdihat←(1,d)⍴0
 dExdfhat←(1,d)⍴0
 dExdohat←(1,d)⍴0

 dzt←(1,d)⍴0
 I←(1,d)⍴0
 dExdWt←(1,d)⍴0
 dExdH←(1,d)⍴((1000?1000)÷1000000) ⍝ random numbers for err derivative, for now

 ⍝ vectorizing all multiplications here
 dExdot←dExdH×(7○ct)
 dExdct←dExdct+dExdH×ot×(1-7○ct)
 dExdit←dExdct×at
 ⍝ dE/dC1 dE/dC2 dE/dC3 dE/dC4 ....
 ⍝ ×      ×      ×      ×
 ⍝ c0     ct1    ct2   ct3     ....
 ⍝(dE/dCt)×(ct-1) , if t=1, then (dE/dCt)×c0
 tmp←c0[;1],ct[;⍳(¯1+(¯1↑⍴ct))]
 dExdft←dExdct×tmp
 dExdat←dExdct×it
 dExdcprev←dExdct×ft

 dExdahat←dExdat×(1-(7○athat)*2)
 dExdihat←dExdit×it×(1-it)
 dExdfhat←dExdft×ft×(1-ft)
 dExdohat←dExdot×ot×(1-ot)
 ⍝ this block combines all arrays and multiplies in one fell swoop
 ⍝ stylish, but difficult to interpret, so I've taken
 ⍝ the simpler approach

 ⍝ dzt←⊂(1,d)⍴(dExdahat dExdihat dExdfhat dExdohat)
 ⍝ dzt←(4 1)⍴(dExdahat dExdihat dExdfhat dExdohat)
 ⍝ tmp←h0[;1],ht[;⍳(¯1+(¯1↑⍴ht))]
 ⍝ I←(2 1)⍴(xt tmp)
 ⍝ ⍝ dExdWt←I+.×⍉dzt ⍝ 1st row delta Ws, 2nd delta Us

 ⍝ Simpler approach, multiply each individual array
 ⍝ delta Ws - (x to a, i, f, o)
 deltaWa←xt+.×dExdahat
 Wa←Wa+deltaWa
 deltaWi←xt+.×dExdihat
 Wi←Wi+deltaWi
 deltaWf←xt+.×dExdfhat
 Wf←Wf+deltaWf
 deltaWo←xt+.×dExdohat
 Wo←Wo+deltaWo

 ⍝ delta Us - (h to a, i, f, o)
 ⍝ outer product here, multiply ht with each derivative - fully connected
 ⍝ each ht is propogated to each other node, so outer product
 deltaUa←+⌿(d,d,d)⍴,tmp∘.×⍉dExdahat
 Ua←Ua+deltaUa
 deltaUi←+⌿(d,d,d)⍴tmp∘.×⍉dExdihat
 Ua←Ui+deltaUi
 deltaUf←+⌿(d,d,d)⍴tmp∘.×⍉dExdfhat
 Uf←Uf+deltaUf
 deltaUo←+⌿(d,d,d)⍴tmp∘.×⍉dExdohat
 Uo←Uo+deltaUo

 ⍝ Now to do the same for hprev
 ⍝ ta←Wa+.×dExdahat
 ⍝ ti←Wi+.×dExdihat
 ⍝ tf←Wf+.×dExdfhat
 ⍝ to←Wo+.×dExdohat

 hprevtoa←+⌿(d,d,d)⍴Ua∘.×⍉dExdahat
 hprevtoi←+⌿(d,d,d)⍴Ui∘.×⍉dExdihat
 hprevtof←+/(d,d,d)⍴Uf∘.×⍉dExdfhat
 hprevtoo←+⌿(d,d,d)⍴Uo∘.×⍉dExdohat

 dhprev←hprevtoa+hprevtof+hprevtoi+hprevtoo
 hprev←hprev+dhprev

 ⍝ All weights and outputs incremented toward convergence.
 ⍝ Now onto gradient clipping





 ⍝ this code below is vectorized above - I'm leaving it here for reference. 
 ⍝ t←numInputUnits
     ⍝:While t≥1
⍝         dExdot[;t]←dExdH[;t]×(7○ct[;t])
⍝
⍝         dExdct[;t]←dExdct[;t]+dExdH[;t]×ot[;t]×(1-(7○ct[;t])*2)
⍝
⍝         dExdit[;t]←dExdct[;t]×at[;t]
⍝         :If t>1
⍝             dExdft[;t]←dExdct[;t]×ct[;t-1]
⍝         :Else
⍝             dExdft[;t]←dExdct[;t]×c0[;t]
⍝         :EndIf
⍝         dExdat[;t]←dExdct[;t]×it[;t]
⍝         dExdcprev←dExdct[;t]×ft[;t]
⍝
⍝         ⍝ sigmoid derivatives
⍝         dExdahat←dExdat[;t]×(1-(7○athat[;t])*2)
⍝         dExdihat←dExdit[;t]×it[;t]×(1-it[;t])
⍝         dExdfhat←dExdft[;t]×ft[;t]×(1-ft[;t])
⍝         dExdohat←dExdot[;t]×ot[;t]×(1-ot[;t])
⍝         dzt[;t]←⊂(1,d)⍴(dExdahat dExdihat dExdfhat dExdohat)
⍝
⍝         :If t>1
⍝             I[;t]←⊂(2 1)⍴((xt[t;])(ht[;t-1]))
⍝         :Else
⍝             I[;t]←⊂(2 1)⍴((xt[t;])(h0[;t]))
⍝         :EndIf
⍝         dExdWt[;t]←⊂(⍉↑dzt[;t])+.×⍉↑I[;t]
⍝         t←t-1
⍝     :EndWhile
