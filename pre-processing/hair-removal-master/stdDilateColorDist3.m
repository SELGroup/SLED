function [J1]=stdDilateColorDist3(I,J,M,n1,high,s,fflag)
x=[1 1 0 -1 -1 -1  0  1 2 2 2  2  2 1  1 0  0 -1 -1 -2 -2 -2 -2 -2];
y=[0 1 1  1  0 -1 -1 -1 2 1 0 -1 -2 2 -2 2 -2  2 -2  2  1  0 -1 -2];
% s=20;
[n,m]=size(M);
M=double(M);
list=zeros(n*m,5);
nh=double(zeros(n*m,3));
cform = makecform('srgb2lab');
lab = applycform(I,cform); 
J1=J;
idx=1;
for i=1:n
    for j=1:m
        if M(i,j)==1
            idx=idx+1;
            count=1;
            ptr=1;
            M(i,j)=idx;
            list(count,1)=i;
            list(count,2)=j;
            count=count+1;
            % build seeds
            while ptr<count
                ix=list(ptr,1);
                iy=list(ptr,2);
                ptr=ptr+1;
                for a=1:8
                    ix1=ix+x(a);
                    iy1=iy+y(a);
                    if ix1>=1 & ix1<=n & iy1>=1 & iy1<=m
                        if M(ix1,iy1)==1
                            M(ix1,iy1)=idx;
                            list(count,1)=ix1;
                            list(count,2)=iy1;
                            list(count,3)=double(lab(ix1,iy1,1));
                            list(count,4)=double(lab(ix1,iy1,2));
                            list(count,5)=double(lab(ix1,iy1,3));
                            count=count+1;
                        end
                    end
                end
            end

            if count<=2 %low
                idx=idx-1;
                for k=1:count-1
                    M(list(k,1),list(k,2))=0;
                    J1(list(k,1),list(k,2),:)=255;
                end

            else % grow from the color closest pixel
                ptr=0;
                for k=1:count-1 % build nh
                    ix=list(k,1);
                    iy=list(k,2);
                    for a=1:8
                        ix1=ix+x(a);
                        iy1=iy+y(a);
                        if ix1>=1 & ix1<=n & iy1>=1 & iy1<=m
                            if M(ix1,iy1)<1
                                M(ix1,iy1)=idx;
                                x0=ix1-s;
                                x1=ix1+s;
                                y0=iy1-s;
                                y1=iy1+s;
                                if x0<1
                                    x0=1;
                                end
                                if x1>n
                                    x1=n;
                                end
                                if y0<1
                                    y0=1;
                                end
                                if y1>m
                                    y1=m;
                                end
                                dist=findLdaDist(M(x0:x1,y0:y1),double(lab(x0:x1,y0:y1,:)),double([lab(ix1,iy1,1);lab(ix1,iy1,2);lab(ix1,iy1,3)]));
                                ptr=ptr+1;
                                nh(ptr,:)=[dist ix1 iy1];
                            end
                        end
                    end
                end
                if ptr>=1
                    dlen=0;
                    mn=min(nh(1:ptr,1));
                    n0=count/2;
                    while dlen<=n0 & mn<=high % find the closest in color from nh
                        dlen=dlen+1;
                        [mn,pos]=min(nh(1:ptr,1));
                        ix=nh(pos,2);
                        iy=nh(pos,3);
                        nh(pos,1)=255;
                        list(count,1)=ix; % put the point in the line region
                        list(count,2)=iy;
                        J1(ix,iy,:)=200;
                        count=count+1;
                        for a=1:8 % expand nh for the added point
                            ix1=ix+x(a);
                            iy1=iy+y(a);
                            if ix1>=1 & ix1<=n & iy1>=1 & iy1<=m
                                if M(ix1,iy1)<=1
                                    M(ix1,iy1)=idx;
                                    x0=ix1-s;
                                    x1=ix1+s;
                                    y0=iy1-s;
                                    y1=iy1+s;
                                    if x0<1
                                        x0=1;
                                    end
                                    if x1>n
                                        x1=n;
                                    end
                                    if y0<1
                                        y0=1;
                                    end
                                    if y1>m
                                        y1=m;
                                    end
                                    dist=findLdaDist(M(x0:x1,y0:y1),double(lab(x0:x1,y0:y1,:)),double([lab(ix1,iy1,1);lab(ix1,iy1,2);lab(ix1,iy1,3)]));
                                    ptr=ptr+1;
                                    nh(ptr,:)=[dist ix1 iy1];
                                end
                            end
                        end
                        mn=min(nh(1:ptr,1));
                    end
                end

                for k=1:ptr % clear nh relieve un-used
                    if nh(k,1)<255
                        M(nh(k,2),nh(k,3))=0;
                    end
                end
                if count<6 % remove small region
                    for k=1:count-1
                        M(list(k,1),list(k,2))=0;
                        J1(list(k,1),list(k,2),:)=255;
                    end
                else % grow by one pixel
                    if fflag>0
                        for a=1:count-1
                            ix=list(a,1);
                            iy=list(a,2);
                            for k=1:8
                                ix1=ix+x(k);
                                iy1=iy+y(k);
                                if ix1>=1 & ix1<=n & iy1>=1 & iy1<=m
                                    if M(ix1,iy1)<=1
                                        M(ix1,iy1)=idx;
                                        J1(ix1,iy1,:)=200;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function [dist]=findAveDist(list,c)
num=size(list,1);
dist=zeros(num,1);
for i=1:num
    dist(i)=((c(1)-list(i,1))^2+(c(2)-list(i,2))^2+(c(3)-list(i,3))^2)^0.5;
end
dist=0.5*median(dist)+0.5*min(dist);

function [dist]=findLdaDist(M,Lab,x0)
m1=zeros(3,1);
m0=zeros(3,1);
x=zeros(3,1);
S1=zeros(3,3);
S0=zeros(3,3);
[m,n]=size(M);
n1=0;
n0=0;
% find mean m1, m0
for i=1:2:m
    for j=1:2:n
        if M(i,j)>0
            n1=n1+1;
            m1(1)=m1(1)+Lab(i,j,1);
            m1(2)=m1(2)+Lab(i,j,2);
            m1(3)=m1(3)+Lab(i,j,3);
        else
            n0=n0+1;
            m0(1)=m0(1)+Lab(i,j,1);
            m0(2)=m0(2)+Lab(i,j,2);
            m0(3)=m0(3)+Lab(i,j,3);
        end
    end
end
m1=m1/n1;
m0=m0/n0;
% find S1, S0
for i=1:2:m
    for j=1:2:n
        x(1)=Lab(i,j,1);
        x(2)=Lab(i,j,2);
        x(3)=Lab(i,j,3);
        if M(i,j)>0
            x=x-m1;
            S1=S1+x*x';
        else
            x=x-m0;
            S0=S0+x*x';
        end
    end
end
% find LDA w
w=inv(S1+S0)*(m1-m0);
t0=(w')*m0;
t1=(w')*m1;
t=(w')*x0;
if t0==t1
    dist=0;
else
    dist=(t-t1)/(t0-t1);
end