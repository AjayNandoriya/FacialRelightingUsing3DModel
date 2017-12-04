addpath(genpath('3DMM_edges'));

% YOU MUST set this to the base directory of the Basel Face Model
BFMbasedir = 'C:\Users\qcri\Documents\Ajay\Face\SOA\3DMM_edges-master\';

% Load morphable model
load(strcat(BFMbasedir,'01_MorphableModel.mat'));
% Important to use double precision for use in optimisers later
shapeEV = double(shapeEV);
shapePC = double(shapePC);
shapeMU = double(shapeMU);
texPC = double(texPC/256);
texMU = double(texMU/256);




% Number of model dimensions to use
ndims = 60;
% Prior weight for initial landmark fitting
w_initialprior=0.7;

test_fname = 'ss1.PNG';
ref_fname = 'ref6.PNG';
outfname = [test_fname(1:end-4) '_' ref_fname(1:end-4) '_output.png'];



img = im2double(imread(test_fname));


img_gt = im2double(imread(ref_fname));


%load shape data
load([test_fname(1:end-3) 'mat']);
% b = csvread('/home/qcri/Documents/ajay/CNN/Face/SOA/3dmm_cnn/demoCode/mean_out/mean_004_1_15.ply.alpha');
% b=b(1:ndims);
FV.vertices=reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC,1)/3)';
FV.faces = tl;
pts2D = pts3Dto2D(FV.vertices,R,t,s)';

FV.visiblePoints = visiblevertices(FV,R);
FV.normals = vertex_normals(FV,R,t,s);


C = [0.429043 0.511664 0.743125 0.886227 0.247708];
H = [C(4)*ones(length(FV.normals),1) 2*C(2)*FV.normals(:,2) 2*C(2)*FV.normals(:,3) 2*C(2)*FV.normals(:,1) 2*C(1)*FV.normals(:,1).*FV.normals(:,2) 2*C(1)*FV.normals(:,2).*FV.normals(:,3) C(3)*FV.normals(:,3).*FV.normals(:,3)-C(5) 2*C(1)*FV.normals(:,1).*FV.normals(:,3) C(1)*(FV.normals(:,1).^2-FV.normals(:,2).^2)];
H = reshape(permute(repmat(H,[1 1 3]),[3 1 2]),[],size(H,2));
H(:,10:27)=0;
H(2:3:end,10:18)=H(2:3:end,1:9);
H(3:3:end,19:27)=H(3:3:end,1:9);
H(2:3:end,1:9)=0;
H(3:3:end,1:9)=0;

cond1 = load([test_fname(1:end-4) '_coeff.mat']);
cond2 = load([ref_fname(1:end-4) '_coeff.mat']);
tex_val = texPC(:,1:ndims)*cond1.texb+texMU;
beta_i = tex_val.*H;
Ir = reshape(beta_i*cond1.L,3,size(texPC,1)/3)';
Ir(Ir<0)=0;
FV.facevertexcdata = Ir;
Ir1 = renderFace(FV,img,R,t,s,true);

tex_val = texPC(:,1:ndims)*cond1.texb+texMU;
beta_i = tex_val.*H;
Ir = reshape(beta_i*cond2.L,3,size(texPC,1)/3)';
Ir(Ir<0)=0;
FV.facevertexcdata = Ir;
Ir2 = renderFace(FV,img,R,t,s,true);


Ir1 = imfilter(Ir1,fspecial('gaussian',7,3),'symmetric');
Ir2 = imfilter(Ir2,fspecial('gaussian',7,3),'symmetric');

I_out = (Ir2.*(img+0.01))./(Ir1+0.01);
imwrite(I_out,outfname);

subplot(122);imshow(I_out(:,:,1));title('output');
subplot(121);imshow(img);title('input');

