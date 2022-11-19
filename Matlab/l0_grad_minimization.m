function [u,h,v] = l0_grad_minimization(y,L)
    N = size(y,1);
    M = size(y,2);
    C = size(y,3);
    
    u = 1.0*y;
    
    h = zeros(size(y));
    v = zeros(size(y));
    
    p = zeros(size(y));
    q = zeros(size(y));
    
    fy = zeros(size(y));
    for c = 1:C
        fy(:,:,c) = fft2(y(:,:,c));
    end
    
    
    dx = zeros([size(y,1),size(y,2)]);
    dy = zeros([size(y,1),size(y,2)]);
    dx(1,1) = -1.0; dx(N,1) =  1.0;
    dy(1,1) = -1.0; dy(1,M) =  1.0;
    
    fdxt = repmat(conj(fft2(dx)),[1,1,size(y,3)]);
    fdyt = repmat(conj(fft2(dy)),[1,1,size(y,3)]);
    
    adxy = abs(fdxt).^2+abs(fdyt).^2;
    
    beta = 0.5/L;
    for t = 1:50
        if beta <= 1e-2
            break;
        end
        disp(t);
        disp(beta);
        
        [ux,uy] = gradients(u);
        
        ls = abs(ux).^2+abs(uy).^2 >= beta*L;
        h = ls.*ux;
        v = ls.*uy;
        
        fh = zeros(size(y));
        fv = zeros(size(y));
        for c = 1:C
            fh(:,:,c) = fft2(h(:,:,c));
            fv(:,:,c) = fft2(v(:,:,c));
        end
        
        fu = (beta*fy+(fdxt.*fh+fdyt.*fv))./(beta+adxy);
        for c = 1:C
            u(:,:,c) = real(ifft2(fu(:,:,c)));
        end
        
        %[ux,uy] = gradients(u);
        %p = p + ux;
        %v = q + uy;
        
        beta = 0.65 * beta;
    end
    
end