
% Solving the 2-D Laplace's equation in a rectangoular domain by the Finite Difference
% Numerical scheme used is a second order central difference in space (5-point difference)

	
		
nx=60;                                   %Number of point in x direction
ny=60;                                   %Number of steps in y direction      
niter=1000;                              %Number of iterations 
dx=2/(nx-1);                    	 %grid spacing x and y
x=(0:dx:2);                       
y=(0:dy:2); 
                      

	
	                       		 %Initial Conditions
p=zeros(ny,nx);                  

		
					%Boundary conditions
p(:,1)=sin(pi*x);			%Dirichlet BC
p(:,nx)=0;				%Dirichlet BC
p(1,:)=0;                      		%Neumann BC
p(ny,:)=0;              		%Neumann BC

				
					%Explicit iterative scheme 
j=2:nx-1;
i=2:ny-1;
for it=1:niter
    pn=p;
    p(i,j)=((dy^2*(pn(i+1,j)+pn(i-1,j)))+(dx^2*(pn(i,j+1)+pn(i,j-1))))/(2*(dx^2+dy^2));
    		
    p(:,1)=sin(pi*x);
    p(:,nx)=y;
    p(1,:)=2;
    p(ny,:)=-x;   
end

%Plot the results

surf(x,y,p,'EdgeColor','none');       
shading interp
title({'2-D Laplace''s equation';['{\itNumber of iterations} = ',num2str(it)]})
xlabel('Spatial co-ordinate (x) \rightarrow')
ylabel('{\leftarrow} Spatial co-ordinate (y)')
zlabel('Solution profile (P) \rightarrow')
pause()
