
% Parabolic eqution in 2 spatial dimension in a rectangoular domain


mu=1; 			%diffusivity
nx=10;  		%number node x
ny=10; 			%number node y
nt=2000;		%number of time step
a=1; 			%size of the domain
b=1;


T=2; 			%Temporal domain
dx=a/(nx-1);
dy=b/(ny-1);
dt=T/(nt-1);
x=(0:dx:a);
y=(0:dy:b);
t=(0:dt:T);
fo=mu*dt/(dx^2); 	%Lattice Fourier number 		

u=zeros(nt,nx,ny);

			% Initial condition
[X,Y]=meshgrid(x,y);
Z=sin(4*pi*X).*cos(4*pi*Y);
u(1,:,:)=Z;

			%Boundary conditions
for k=1:1:nt
	u(k,1,1:ny)=0;
	u(k,1:nx,1)=0;
	u(k,1:nx,ny)=0;
	u(k,nx,1:ny)=0;
end


for k=1:1:nt
	for i=2:1:nx-1
		for j=2:1:ny-1
u(k+1,i,j)=(1-4*fo)*u(k,i,j)+fo*(u(k,i-1,j)+u(k,i+1,j)+u(k,i,j-1)+u(k,i,j+1));
		end
	end
end


			% Plotting the result at different time steps

a=zeros(nx,ny);
for i=1:nx;
	for j=1:ny
		a(i,j)=u(1,i,j);
		b(i,j)=u(nt/200,i,j);
		c(i,j)=u(nt/100,i,j);
		d(i,j)=u(nt/50,i,j);
		e(i,j)=u(nt/10,i,j);
		f(i,j)=u(nt,i,j);
		
	end
end


subplot(3,2,1)
surf(x,y,a);
subplot(3,2,2)
surf(x,y,b);
subplot(3,2,3)
surf(x,y,c);
subplot(3,2,4)
surf(x,y,d);
subplot(3,2,5)
surf(x,y,e);
subplot(3,2,6)
surf(x,y,f);
pause()


			


    
