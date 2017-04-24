%function [x,t,u]=alembert()
	% risoluzione dell equazione d alembert
	%metodo differenze finite
L=20;
T=5;
nx=101;
nt=101;
c=1 ;      % velocita di propagazione unitaria
dx=L/(nx-1);
dt=T/(nt-1);
x=(-L/2: dx:L/2);
t=(0:dt:T);
u=zeros(nt,nx);
a=dt/dx;

%condizioni iniziali su u e sulla derivata prima rispetto al tempo
p=zeros(1,nx);
p=exp(-x.^2);
u(1,:)=p;
% condizione ai bordi
%u(:,1)=0;
%u(:,nx)=0;
% derivata prima nulla

for j=2:1:nx-1
	
	u(2,j)=0.5*a^2*(p(1,j-1)+p(1,j+1))+(1-a^2)*p(1,j);

end

% implementazione codice

for i=2:1:nt
	for j=2:1:nx-1
	
		u(i+1,j)=a^2*u(i,j-1)+2*(1-a^2)*u(i,j)+a^2*u(i,j+1)-u(i-1,j);
	end
	plot(x,u(i,:));
	pause(0.1);
end

pause()


