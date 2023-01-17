function [I1]=HairRemovMed(I,M,s)
[m,n]=size(M);
global r;
global g;
global b;
global total;
r=zeros((s+1)^2,1);
g=zeros((s+1)^2,1);
b=zeros((s+1)^2,1);
I1=I;
for i=1:m
    for j=1:n
        if M(i,j)<1
            x0=i-s;
            x1=i+s;
            y0=j-s;
            y1=j+s;
            if x0<1
                x0=1;
            end
            if x1>m
                x1=m;
            end
            if y0<1
                y0=1;
            end
            if y1>n
                y1=n;
            end
            setRGB(I(x0:x1,y0:y1,:),M(x0:x1,y0:y1),x1-x0+1,y1-y0+1);
            I1(i,j,1)=median(r(1:total));
            I1(i,j,2)=median(g(1:total));
            I1(i,j,3)=median(b(1:total));
        end
    end
end

function setRGB(I,M,m,n)
global r;
global g;
global b;
global total;
total=0;
for i=1:2:m
    for j=1:2:n
        if M(i,j)>0
            total=total+1;
            r(total)=I(i,j,1);
            g(total)=I(i,j,2);
            b(total)=I(i,j,3);
        end
    end
end