function [Ix,Iy] = gradients(I)
    N = size(I,1);
    M = size(I,2);
    
    Ix = zeros(size(I));
    Iy = zeros(size(I));

    for i = 1:N
        for j = 1:M
            if i < N
                Ix(i,j,:) = I(i+1,j,:)-I(i,j,:);
            else
                Ix(N,j,:) = I(  1,j,:)-I(N,j,:);
            end
            
            if j < M
                Iy(i,j,:) = I(i,j+1,:)-I(i,j,:);
            else
                Iy(i,M,:) = I(i,  1,:)-I(i,M,:);
            end
        end
    end
end