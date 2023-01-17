function [J]=stdDilateDarkest(I,high,low,n1)
x=[1 1 0 -1 -1 -1  0  1 2 2 2  2  2 1  1 0  0 -1 -1 -2 -2 -2 -2 -2];
y=[0 1 1  1  0 -1 -1 -1 2 1 0 -1 -2 2 -2 2 -2  2 -2  2  1  0 -1 -2];
[n,m,c]=size(I);
list=zeros(n*m,2);
nh=double(zeros(n*m,3));
M=zeros(n,m);
J=uint8(zeros(n,m,3));
for i=1:n
    for j=1:m
        if I(i,j,1)>=high
            M(i,j)=1;
        else
            J(i,j,:)=255;
        end
    end
end
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
                            count=count+1;
                        end
                    end
                end
            end
            if count<=2 %low
                idx=idx-1;
                for k=1:count-1
                    M(list(k,1),list(k,2))=0;
                    J(list(k,1),list(k,2),:)=255;
                end
            else % grow from the most-line-like
                ptr=0;
                for k=1:count-1 % build nh
                    ix=list(k,1);
                    iy=list(k,2);
                    for a=1:8
                        ix1=ix+x(a);
                        iy1=iy+y(a);
                        if ix1>=1 & ix1<=n & iy1>=1 & iy1<=m
                            if M(ix1,iy1)<=1 & I(ix1,iy1,1)>=low
                                M(ix1,iy1)=idx;
                                ptr=ptr+1;
                                nh(ptr,:)=[double(I(ix1,iy1,1)) ix1 iy1];
                            end
                        end
                    end
                end
                if ptr>=1
                    dlen=0;
                    mx=max(nh(1:ptr,1));
                    
                    while dlen<=n1 & mx>=low % find the line-like-most from nh
                        dlen=dlen+1;
                        [mx,pos]=max(nh(1:ptr,1));
                        ix=nh(pos,2);
                        iy=nh(pos,3);
                        nh(pos,1)=0;
                        list(count,1)=ix; % put the point in the line region
                        list(count,2)=iy;
                        J(ix,iy,:)=150;
                        count=count+1;
                        for a=1:8 % expand nh for the added point
                            ix1=ix+x(a);
                            iy1=iy+y(a);
                            if ix1>=1 & ix1<=n & iy1>=1 & iy1<=m
                                if M(ix1,iy1)<=1 & I(ix1,iy1,1)>=low
                                    M(ix1,iy1)=idx;
                                    ptr=ptr+1;
                                    nh(ptr,:)=[double(I(ix1,iy1,1)) ix1 iy1];
                                end
                            end
                        end
                        mx=max(nh(1:ptr,1));
                    end
                end
                for k=1:ptr % clear nh relieve un-used
                    if nh(k,1)>0
                        M(nh(k,2),nh(k,3))=0;
                    end
                end
                if count<12 % remove small region
                    for k=1:count-1
                        M(list(k,1),list(k,2))=0;
                        J(list(k,1),list(k,2),:)=255;
                    end
                else % remove not line-like enough
                    temp=zeros(count-1,1);
                    for k=1:count-1
                        ix=list(k,1);
                        iy=list(k,2);
                        temp(k)=double(I(ix,iy,1));
                    end
                    [B,IX]=sort(temp,'descend');
                    if B(10)<0.75*255
                        if median(temp)<0.725*255
                            for k=1:count-1
                                M(list(k,1),list(k,2))=0;
                                J(list(k,1),list(k,2),:)=255;
                            end
                        end
                    end
                end
            end
        end
    end
end
% for i=1:n
%     for j=1:m
%         if J(i,j,1)>=1 & M(i,j)>1
%             J(i,j,:)=150;
%         end
%     end
% end