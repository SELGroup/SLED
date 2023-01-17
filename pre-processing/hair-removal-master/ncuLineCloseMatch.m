function [J]=ncuLineCloseMatch(I,t)
[m,n]=size(I);
J=zeros(m,n);
for deg=-90:180/t:90-180/t
	Ji=ncuLineMatchFilterDirec(I,deg);
    for i=1:m
        for j=1:n
            if Ji(i,j)>J(i,j)
                J(i,j)=Ji(i,j);
            end
        end
    end
end 