
%2D_Poisson equation L(p)=f in a rectangoular domain

Lx=2;
Ly=2;

nx=20; 				%nodes in x
ny=20;				%nodes in y

niter=1000;  			%max number of pseudo temporal iterations

dx=Lx/(nx-1);
dy=Ly/(ny-1);
x=(0:dx:Lx);
y=(0:dy:Ly);
pn=zeros(ny,nx);

				%Initialization
p=zeros(ny,nx);

				%Source Term
[X,Y]=meshgrid(x,y);
Z=sin(X).*cos(Y);
s=zeros(ny,nx);
s(:,:)=Z;




%Dirichlet Boundary conditions 

p(:,1)=0;
p(:,nx)=0;
p(1,:)=-0;                      
p(ny,:)=0;


j=2:1:nx-1;
i=2:1:ny-1;

for k=1:1:niter
	pn=p;
p(i,j)=((dy^2*(pn(i+1,j)+pn(i-1,j)))+(dx^2*(pn(i,j+1)+pn(i,j-1)))+(s(i,j)*dx^2*dy^2))/(2*(dx^2+dy^2));
	
	p(:,1)=0;
	p(:,nx)=0;
	p(1,:)=0;                      
	p(ny,:)=0;
end

surf(x,y,p,'EdgeColor','none');       
shading interp;
pause()

	
