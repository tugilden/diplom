from sympy import *
import math as m


def evklid(a,b):
    x, y, b, d = 1, 0, abs(b), b
    while a != 0:
        a,d,x,y=d%a,a,y-x*(d//a),x
    if d<0:
        d=-d;y=-y
    y=y%(b//d)
    return y, d


def det(a,b):
    return a[0]*b[1]-a[1]*b[0]

def get_abc(n):
    a, b, c = 2, 1, 0
    for _ in range(n-2):
        a, b, c = 2*a+b, a, b
    return a, b, a - b // 2 - 1

def AngleHull(a,b,c):
    P = Matrix([c // a, 0])
    c = c % a
    p = P
    h1, h2 = Matrix([1,0]), Matrix([0,1])
    H = Matrix([[], []])
    w1, w2 = a, b
    while True:
        if w1 <= c: 
            t1 = c // w1
            p += t1*h1
            P = P.col_insert(0,p)
            c -= t1 * w1
            H = H.col_insert(0,h1)
            if c == 0:
                return P, H
        if w2 == 0:
            if c == 0:
                return P, H
            if w1 <= c and w1 != 0:
                t1 = c // w1
                p += t1 * h1
                P = P.col_insert(0, p)
                H = H.col_insert(0, h1)
                c -= t1 * w1
                if c == 0:
                    return P, H
            return P, H
        t = m.ceil(w1 / w2)
        if t >= 3:
            h1, h2 = h2, t * h2 - h1
            w1, w2 = w2, t * w2 - w1
        else:
            g = h2 - h1
            d = w2 - w1
            s = w1 // (w1 - w2)
            v=m.ceil((c-w1)/d)
            if v<=s-1:
                h=h1+v*g 
                w=w1+v*d                
                t1=c//w
                p+=t1*h
                P=P.col_insert(0,p)
                H=H.col_insert(0,h)  
                c-=t1*w
                if c==0:
                    return P,H  
            h2=h1+s*g    
            w2=w1+s*d
            if w1==0:
                p+=t1*h1
                c-=t1*w1
                P=P.col_insert(0,p)
                H=H.col_insert(0,h1)  
                return P,H
            h1=h2-g
            w1=w2-d
            if w2 == 0:
                if c == 0:
                    return P, H
                if w1 <= c and w1 != 0:
                    t1 = c // w1
                    p += t1 * h1
                    P = P.col_insert(0, p)
                    H = H.col_insert(0, h1)
                    c -= t1 * w1
                    if c == 0:
                        return P, H
                return P, H
            t=m.ceil(w1/w2)
            h1,h2=h2,t*h2-h1
            w1,w2=w2,t*w2-w1
            if w1<=c: 
                t1=c//w1
                p+=t1*h1
                P=P.col_insert(0,p)
                c-=t1*w1
                H=H.col_insert(0,h1)
            if c==0:
                return P,H
            if w2==0:
                print(h1)
                return P,H

def Grahem(a, b, c):                
    print(evklid(a, b))
    y, _ = evklid(a, b)
    t=(c*y)%b-b
    print('t=',t)
    S = Matrix([[i,(c-i*a)//b] for i in range(0,t-1,-1)])
    V = Matrix([[0,0]])
    S=S.row_insert(0,V)

    i=0
    while i<S.rows-2:
        if det(-S[i,:]+S[i+1,:],-S[i,:]+S[i+2,:])<=0:
            S.row_del(i+1)
            if i!=0:i-=1
        else:
            i+=1
    
    return S

__all__ = ['Grahem', 'AngleHull', 'evklid', 'det']