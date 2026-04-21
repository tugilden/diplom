# Вершины углового полиэдра находятся двумя алгоритмами.
# 1.Проход по "верхним" точкам методом Грэхема
# 2.Основным алгоритмом

from sympy import *
import math as m


def evklid(a,b):
    """
    Вычисляет решение уравнения a*y=gcd(a,b) mod b 
    """
    x, y, b, d = 1, 0, abs(b), b
    while a != 0:
        a,d,x,y=d%a,a,y-x*(d//a),x
    if d<0:
        d=-d;y=-y
    y=y%(b//d)
    return y, d # d=gcd(a,b)>0

def det(a,b):
    return a[0]*b[1]-a[1]*b[0]

def get_abc(n):
    """
    генератор чисел beta_n,beta_{n-1},gamma_n - коэффициентов неравенства с n вершинами в выпуклой оболочке
    """
    a, b, c = 2, 1, 0
    for _ in range(n-2):
        a, b, c = 2*a+b, a, b
    return a, b, a - b // 2 - 1 # a,b,c= beta_n,beta_{n-1},gamma_n=beta_n-beta_{n-1}//2-1

"""
Функция AngleHull(a,b,c)
Вершины выпуклой оболочки целых решений системы неравенств
 a*x1+b*x2<=c, -x2<=0
"""
def AngleHull(a,b,c):
    P = Matrix([c // a, 0]) # в массив P помещена вершина на прямой x2=0
    c = c % a
    p = P
    h1, h2 = Matrix([1,0]), Matrix([0,1])
    H = Matrix([[], []])
    w1, w2 = a, b
    while True:
        #input()
        # Проверяем допустим ли вектор h1. 
        # Если да(w1<=c), то вычисляем очередную вершину.
        if w1 <= c: 
            t1 = c // w1
            p += t1*h1
            P = P.col_insert(0,p)
            c -= t1 * w1
            H = H.col_insert(0,h1)
            if c == 0: # найдена вершина на прямой a*x1+b*x2=c.
                #print('Выход в строке 51')
                return P, H
        # начинаем формировать новый конус
        t = m.ceil(w1 / w2) # ближайшее целое сверху
        if t >= 3:
            h1, h2 = h2, t * h2 - h1 # найден новый конус (h1,h2) в случае t>=3. Переходим в начало цикла while
            w1, w2 = w2, t * w2 - w1
#######################################################################################################       
        else: # появилась прогрессия с разностью g=h2-h1. Вес вектора g равен w2-w1
            g = h2 - h1
            d = w2 - w1
            s = w1 // (w1 - w2) # s+1 -длина прогрессии h1,h1+g,..h1+sg. Веса элементов прогрессии должны быть 
                          # неотрицательны! Следовательно, a(h1+sg)>=0 -> w1+s(w2-w1)>=0 -> s<=w1/(w1-w2)              
            v=m.ceil((c-w1)/d) # 
            #print('64 v=',v,'s-1=',s-1,'c=',c)
            if v<=s-1: # внутри прогрессии есть допустимый вектор h=h1+v*g с весом w=w1+v*d              
                h=h1+v*g 
                w=w1+v*d                
                t1=c//w
                #print('69 t1=',t1, 'w=',w)
                p+=t1*h
                
                P=P.col_insert(0,p)
                H=H.col_insert(0,h)  
                #print('строка 71')
                #pprint(P)
                #pprint(H)
                c-=t1*w
                #print('84 c=',c)
                if c==0: # найдена вершина на прямой a*x1+b*x2=c.
                    #print('Выход в строке 74')
                    return P,H  
            # вычисляем новый конус h1 - последний вектор прогрессии
            # h2- первый вектор после прогрессии h2<--ceil(w_s/w_{s-1})h_s-h_{s-1} 
            # 
            
            h2=h1+s*g    
            w2=w1+s*d
            #print('94 h w=',h,w2)
            #print('94 g d=',g,d)
            if w1==0:
                #print('Выход в строке 82')
                p+=t1*h1
                c-=t1*w1
                #print('100 c=',c)
                P=P.col_insert(0,p)
                H=H.col_insert(0,h1)  
                return P,H
            h1=h2-g      # предпоследний вектор прогрессии
            w1=w2-d
            #print('105 h1, w1',h1,w1)
            t=m.ceil(w1/w2)
            
            ##########
            
            h1,h2=h2,t*h2-h1 # 
            w1,w2=w2,t*w2-w1
            if w1<=c: 
                t1=c//w1
                p+=t1*h1
                P=P.col_insert(0,p)
                c-=t1*w1
                H=H.col_insert(0,h1)
            if c==0: # найдена вершина на прямой a*x1+b*x2=c.
                #print('Выход в строке 51')
                return P,H
            if w2==0:
                print(h1)
                #print('Выход в строке 91')
                return P,H
             # найден новый конус (h1,h2) в случае t=2. Переходим в начало цикла while  

"""
Анализ "верхних" точек методом Грэхема
"""
def Grahem(a, b, c):                
    print(evklid(a, b))
    y, _ = evklid(a, b)
    t=(c*y)%b-b
    print('t=',t)
    S = Matrix([[i,(c-i*a)//b] for i in range(0,t-1,-1)])
    V = Matrix([[0,0]])
    S=S.row_insert(0,V)

    i=0
    while i<S.rows-2: # обход Грэхема
        #print(i)
        if det(-S[i,:]+S[i+1,:],-S[i,:]+S[i+2,:])<=0:
            S.row_del(i+1)
            if i!=0:i-=1
            #print(S,S.rows,i)
        else:
            i+=1
            #print(i)
    
    return S


# Функции для работы с вершинами углового полиэдра
# Экспортируются для использования в main.py

# Дополнительно экспортируем полезные функции
__all__ = ['Grahem', 'AngleHull', 'evklid', 'det']
