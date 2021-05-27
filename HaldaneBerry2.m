

phi=3*pi/2;
t1=1;
t2=.1;
M=t2*3*sqrt(3)*sin(phi)-.1;


%nearest neighbor vectors
a1=[1/2 sqrt(3)/2];
a2=[-1 0];
a3=[1/2 -sqrt(3)/2];

%next nearest neighbor set
b1=a2-a3;
b2=a3-a1;
b3=a1-a2;

%reciprocal lattice vectors
c1=[4*pi/3 0];
c2=[2*pi/3 2*pi/sqrt(3)];

%reciprocal vector coefficients defining the unit cell
dlt=0.005;
u10=0:dlt:1;
u20=u10;

[u1, u2]=meshgrid(u10,u20);

kx=u1*c1(1)+u2*c2(1);
ky=u1*c1(2)+u2*c2(2);

lowerband=zeros(size(kx));
upperband=zeros(size(kx));
berrycurve=zeros(size(kx));

h=0.0001;

for xcnt=1:length(u10),
    
    for ycnt=1:length(u20);
        
        k= [kx(xcnt,ycnt); ky(xcnt,ycnt)];
        
        cosasum=cos(a1*k)+cos(a2*k)+cos(a3*k);
        sinasum=sin(a1*k)+sin(a2*k)+sin(a3*k);
        
        cosbsum=cos(b1*k)+cos(b2*k)+cos(b3*k);
        sinbsum=sin(b1*k)+sin(b2*k)+sin(b3*k);
        
        H=zeros(2);
        H(1,1)=M+2*t2*cos(phi)*cosbsum-2*t2*sin(phi)*sinbsum;
        H(2,2)=-M+2*t2*cos(phi)*cosbsum+2*t2*sin(phi)*sinbsum;
        H(1,2)=t1*(cosasum-1i*sinasum);
        H(2,1)=conj(H(1,2));
        
        [v0,d0]=eig(H);
        
        u0=v0(:,1);
        u0=conj(u0(2))/abs(u0(2))*u0;
        
        lowerband(xcnt,ycnt)=min(min(d0));
        upperband(xcnt,ycnt)=max(max(d0));
        
        %%%%%%%% x direction
        
        kxx=k+[h; 0];
        
        cosasum=cos(a1*kxx)+cos(a2*kxx)+cos(a3*kxx);
        sinasum=sin(a1*kxx)+sin(a2*kxx)+sin(a3*kxx);
        
        cosbsum=cos(b1*kxx)+cos(b2*kxx)+cos(b3*kxx);
        sinbsum=sin(b1*kxx)+sin(b2*kxx)+sin(b3*kxx);
        
        
        H=zeros(2);
        H(1,1)=M+2*t2*cos(phi)*cosbsum-2*t2*sin(phi)*sinbsum;
        H(2,2)=-M+2*t2*cos(phi)*cosbsum+2*t2*sin(phi)*sinbsum;
        H(1,2)=t1*(cosasum-1i*sinasum);
        H(2,1)=conj(H(1,2));
        
        
        [vx,dx]=eig(H);
        
        ux=vx(:,1);
        ux=conj(ux(2))/abs(ux(2))*ux;
        
        %%%%%%%% y direction
        
        kyy=k+[0; h];
        
        
        cosasum=cos(a1*kyy)+cos(a2*kyy)+cos(a3*kyy);
        sinasum=sin(a1*kyy)+sin(a2*kyy)+sin(a3*kyy);
        
        cosbsum=cos(b1*kyy)+cos(b2*kyy)+cos(b3*kyy);
        sinbsum=sin(b1*kyy)+sin(b2*kyy)+sin(b3*kyy);
        
        
        H=zeros(2);
        H(1,1)=M+2*t2*cos(phi)*cosbsum-2*t2*sin(phi)*sinbsum;
        H(2,2)=-M+2*t2*cos(phi)*cosbsum+2*t2*sin(phi)*sinbsum;
        H(1,2)=t1*(cosasum-1i*sinasum);
        H(2,1)=conj(H(1,2));
        
        
        [vy,dy]=eig(H);
        
        uy=vy(:,1);
        uy=conj(uy(2))/abs(uy(2))*uy;
        
        %%%%%%%  derivatives
        
        xder=(ux-u0)/h;
        yder=(uy-u0)/h;
        
        berrycurve(xcnt,ycnt)=-i*(xder'*yder-yder'*xder);
        
    end,
    
    xcnt,
    
end,

figure(1);clf;
mesh(kx/pi,ky/pi,berrycurve);
        
sumchern=sum(sum(berrycurve(1:length(berrycurve)-1,1:length(berrycurve)-1)))*dlt^2*(4*pi/3)^2*sin(pi/3)/2/pi

title(strcat('Total Chern number=',num2str(sumchern)));
xlabel('k_x/\pi')
ylabel('k_y/\pi')
        
figure(2);clf

 meshc(kx/pi,ky/pi,lowerband);
 title('Lower band fully occupied');
xlabel('k_x/\pi')
ylabel('k_y/\pi')

