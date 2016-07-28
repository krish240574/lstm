 z←lstm_main
 ⍝ http://arunmallya.github.io/writeups/nn/lstm/index.html#/2
 ⍝ size of input vector

 n←10
 ⍝ number of memory cells/hidden units
 d←3

 ⍝ +1 for bias
 xt←((n),1)⍴((1000?1000)÷1000000)
 W←((4×d),(n+d))⍴((1000?1000)÷1000000)

 ⍝ per cell Ws and Us
 Wa←(n,d)⍴((1000?1000)÷1000000)
 Wi←(n,d)⍴((1000?1000)÷1000000)
 Wf←(n,d)⍴((1000?1000)÷1000000)
 Wo←(n,d)⍴((1000?1000)÷1000000)

 Ua←(d,d)⍴((1000?1000)÷1000000)
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

 ⍝ Forward pass
 ⍝ :While oloop≤10
 t←1 ⍝ time t
 xt←((n),1)⍴((1000?1000)÷1000000)
 :While t≤numInputUnits
      ⍝ node t gets xt, t+1 gets xt+1...
     x←(1 1)⍴xt[t;]
     :If t=1
         cprev←c0
         hprev←h0
     :Else
         cprev[;t]←ct[;t-1]
         hprev[;t]←ht[;t-1]
     :EndIf

     athat[;t]←+⌿((⍉Wa)+.×x)+(Ua+.×⍉hprev)
     at[;t]←7○athat[;t]

     ithat←+⌿((⍉Wi)+.×x)+(Ui+.×⍉hprev)
     it[;t]←1÷(1+*(¯1×ithat))

     fthat←+⌿((⍉Wf)+.×x)+(Uf+.×⍉hprev)
     ft[;t]←1÷(1+*(¯1×fthat))

     othat←+⌿((⍉Wo)+.×x)+(Uo+.×⍉hprev)
     ot[;t]←1÷(1+*(¯1×othat))

     tmp←(it×at)+(ft×cprev)
     ct[;t]←tmp[;t]


     ht[;t]←(ot[;t])×(7○ct[;t])


     t←t+1
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


 dExdot←dExdH×(7○ct)
 dExdct←dExdct+dExdH×ot×(1-7○ct)
 dExdit←dExdct×at
 tmp←c0[;1],ct[;⍳(¯1+(¯1↑⍴ct))]
 dExdft←dExdct×tmp
 dExdat←dExdct×it
 dExdcprev←dExdct×ft

 dExdahat←dExdat×(1-(7○athat)*2)
 dExdihat←dExdit×it×(1-it)
 dExdfhat←dExdft×ft×(1-ft)
 dExdohat←dExdot×ot×(1-ot)
 ⍝dzt←⊂(1,d)⍴(dExdahat dExdihat dExdfhat dExdohat)
 dzt←(4 1)⍴(dExdahat dExdihat dExdfhat dExdohat)
 tmp←h0[;1],ht[;⍳(¯1+(¯1↑⍴ct))]
 I←(2 1)⍴(x tmp)
 dExdWt←I+.×⍉dzt ⍝ 1st row delta Ws, 2nd delta Us

 ⍝⍝ dExdWt←⊂(⍉↑dzt)+.×⍉↑I   ⍝ dUs and dWs, update
⍝ ⍝ Wvec←(1,4)⍴((Wa)(Wi)(Wf)(Wo))
⍝ Wvec←(1 4)⍴(Wa Wi Wf Wo)
⍝ kdzt←(3 3)⍴,⊃dzt
⍝ (Wa+.×kdzt)+(Wi+.×kdzt)+(Wf+.×kdzt)+(Wo+.×kdzt)



 Watmp←(2 1)⍴(Wa Ua)
 Witmp←(2 1)⍴(Wi Ui)
 Wftmp←(2 1)⍴(Wf Uf)
 Wotmp←(2 1)⍴(Wo Uo)

 ⍝ dIt←(⍉W)+.×deXdahat
 ta←Watmp+.×dExdahat
 ti←Witmp+.×dExdihat
 tf←Wftmp+.×dExdfhat
 to←Wotmp+.×dExdohat
 ⍝ hprev to as, is, fs and os of all nodes
 ⍝hprevtoa←+/⊃ta[2;]
⍝ hprevtoi←+/⊃ti[2;]
⍝ hprevtof←+/⊃tf[2;]
⍝ hprevtoo←+/⊃to[2;]
⍝
 hprevtoa←Ua+.×⍉dExdahat
 hprevtoi←Ui+.×⍉dExdihat
 hprevtof←Uf+.×⍉dExdfhat
 hprevtoo←Uo+.×⍉dExdohat
 dhprev←hprevtoa+hprevtof+hprevtoi+hprevtoo

 deltaWs←(d 1)⍴dExdWt[1;]
 deltaUs←(d 1)⍴dExdWt[2;]








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

 sumW←sumW+dExdWt
⍝     oloop←oloop+1
⍝ :EndWhile
