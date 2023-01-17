function [J]=ncuLineMatchFilterDirec(I,deg)
[m,n]=size(I);
I=double(I);
J=zeros(m,n);
deg=(deg)/180*pi;
for k=1:4
    if k==1
        sigma=1;
    elseif k==2
        sigma=sqrt(2);
    elseif k==3
        sigma=2;
    else
        sigma=2*sqrt(2);
    end
    [MF0,s]=G0(sigma,[cos(deg),sin(deg)]);
    a0=sum(sum(MF0.*MF0));
    for i=1:m
        for j=1:n
            flag=0;
            i0=i-s;
            i1=i+s;
            x0=1;
            x1=1+2*s;
            if i0<1
                x0=2+abs(i0);
                i0=1;
                flag=1;
            end
            if i1>m
                x1=x1-(i1-m);
                i1=m;
                flag=1;
            end
            j0=j-s;
            j1=j+s;
            y0=1;
            y1=1+2*s;
            if j0<1
                y0=2+abs(j0);
                j0=1;
                flag=1;
            end
            if j1>n
                y1=y1-(j1-n);
                j1=n;
                flag=1;
            end
            I0=I(i0:i1,j0:j1);
            I0=I0-max(max(I0));
            I0=-I0;
            if flag==1
                MF=MF0(x0:x1,y0:y1);
                a=sum(sum(MF.*MF));
            else
                a=a0;
                MF=MF0;
            end
            b=sum(sum(I0.*I0));
            M0=sum(sum(I0.*MF));
            t=(M0)^2/a/b;
            if t>J(i,j)
                J(i,j)=t;
            end
        end
    end
end

function [F0,d]=G0(sigma,v)
d=ceil(sigma*3);
F0=zeros(2*d+1,2*d+1);
for i=1:2*d+1
    x=i-d-1;
    for j=1:2*d+1
        y=j-d-1;
        t=x*v(1)+y*v(2);
        F0(i,j)=exp(-(x^2+y^2-t^2)/(2*sigma^2));
    end
end