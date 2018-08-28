%y = imread('caustics.tif');
%load('/Users/nick.antipa/Documents/Diffusers/Lensless/3D_calibration/GRIN_pupil_half_degree/Axial/calibration_5micron_20171109_111300/diffuser_miniscope_zstack_20171109.mat');

xin = imread('caustics.tif');
%bgorig = bg_4x;
bgorig = double(imread('~/Documents/Diffusers/Miniscope/bck_4x__H16_M12_S35.png'));
bg_4x = bgorig(:,:,2);

xorig = double(xin(700:(700+size(bgorig,1)-1),1000:(1000+size(bgorig,2)-1)));
%bgorig = bgorig/max(bgorig(:))*max(xorig(:))*.5;

%y = (xorig+bgorig)*.9;
b = bg_4x;
%y = zstack(:,:,16);
yin = double(imread('~/Documents/Diffusers/Miniscope/psf_4x__H16_M3_S47.png'));
y = yin(:,:,2);
alpha = 3.4;
x = 40*ones(size(y));
gx = @(x,alpha)alpha*(alpha*(b+x)-y);
ga = @(x,alpha)alpha*norm(b+x)^2 - sum(sum(y.*(b+x)));

x_step = .01;
alpha_step = .0000000001;
nouter = 100;
ninner = 10;
n = 0;
h1 = figure(1), clf
h2 = figure(2), clf
fprintf('%.2f\t %i\n',norm(alpha*(x+b) - y), alpha)
f = [];
% while n < nouter
%     k = 0;
%     m = 0;
%     n = n+1;
%     fprintf('\n \n \n x \n')
%     while m < ninner
%         m = m + 1;
%         
%         
%         xk = x - x_step * gx(x,alpha);
%         xk = max(soft(xk,tau),0);
%         x = xk;
%         %alpha = alphak;
%         %imagesc(x)
%         %colorbar
%         %drawnow
%         f = cat(1,f,norm(alpha*(x+b) - y)^2);
%         fprintf('%.2f\t %i\n',f(end), alpha)
%     end
%     fprintf('\n\n\n')
%     fprintf('alpha \n')
%     while k < ninner
%         
%         k = k+1;
%         alpha = max(alpha - alpha_step * ga(xk,alpha),0);
%         %imagesc(x)
%         %colorbar
%         %drawnow
%         f = cat(1,f,norm(alpha*(x+b) - y)^2);
%         fprintf('%.2f\t %i\n',f(end), alpha)
%     end
%     if mod(n,1)==0
%         set(0,'CurrentFigure',h1)
%        imagesc(x)
%        colorbar
%        drawnow
%        fprintf('%.2f\t %i\n',norm(alpha*(x+b) - y)^2 + sum(abs(vec(x)))*tau, alpha)
%        set(0,'CurrentFigure',h2)
%        semilogy(f)
%        drawnow
%     end
%     
% end





%% Try ADMM
%b = max(bg_4x(:))*(bg_4x/max(bg_4x(:))).^(.8);
%b = bgorig;
n = 0;
v = ones(size(x));
w = v;
u = w;
niter = 1000;
alpha1 = v;
alpha2 = v;
alpha =1;
alpha3 = v;
mu1 = .7
mu2 = .1
mu3 = .7
tau = 40000/2000;
f = [];
resid_tol = 3e10;
mu_inc = 1.5;
mu_dec = 1.5;
while n < niter
    n = n+1;
    alpha = sqrt(abs(sum(sum(v.*y))/sum(sum(v.*v))));
    uk = u;
    vk = v;
    wk = w;
    xk = x;
    u = soft(x + alpha1/mu1, tau/mu1);
    v = mu2*(b+x) - alpha2 + alpha*y;
    w = max(w+alpha3/mu3,0);
    x = ((mu1*u - alpha1 + mu2*(v-b) + alpha2 + mu3*w - alpha3)/(mu1+mu2+mu3));
    du = uk-u;
    ru = x-u;
    dv = vk - v;
    rv = v-b-x;
    dw = wk - w;
    rw = x-w;
    alpha1 = alpha1 + mu1*ru;
    alpha2 = alpha2 + mu2*rv;
    alpha3 = alpha3 + mu3*rw;
    [mu1, ~] = update_param(mu1,resid_tol,mu_inc,mu_dec,norm(ru,'fro'),norm(mu1*du,'fro'));
    [mu2, ~] = update_param(mu2,resid_tol,mu_inc,mu_dec,norm(rv,'fro'),norm(mu2*dv,'fro'));
    [mu3, ~] = update_param(mu1,resid_tol,mu_inc,mu_dec,norm(rw,'fro'),norm(mu3*dw,'fro'));

    f = cat(1,f,norm(alpha*(x+b) - y)^2);
    
    set(0,'CurrentFigure',h2)
    semilogy(f)
    fprintf('%.2f\t%.2f \t %.2f \t%.2f \t%i\n',f(end), norm(x-u,'fro'),norm(x+b-v,'fro'),norm(w-x,'fro'),alpha)
    drawnow
    
    set(0,'CurrentFigure',h1)
    imagesc(x)
    axis image
    colorbar
    drawnow
end
