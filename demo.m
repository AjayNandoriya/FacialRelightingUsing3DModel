%% Load Data
source_fname = 'ss1.PNG';
ref_fname = 'ref6.PNG';

% Path to 3DMM_edge Repo folder
addpath(genpath('../3DMM_edges'));


% Load model files
load('01_MorphableModel.mat');  
texMU = double(texMU)/255;
texPC = double(texPC)/255;



%% Generate 3d model data
source_img = im2double(imread(source_fname));
ref_img = im2double(imread(ref_fname));

source_img = repmat(source_img(:,:,1),[1 1 3]);
ref_img = repmat(ref_img(:,:,1),[1 1 3]);
[source.normal,source.mask,source.posMap,source.L,source.b,source.texb,source.Ir,FV]=img2facedata(source_img);
[ref.normal,ref.mask,ref.posMap,ref.L,ref.b,ref.texb,ref.Ir]=img2facedata(ref_img);


%% Spherical Harmonic Coefficients
pts2D = pts3Dto2D(FV.vertices,FV.R,FV.t,FV.s)';


C = [0.429043 0.511664 0.743125 0.886227 0.247708];
H = [C(4)*ones(length(FV.normals),1) 2*C(2)*FV.normals(:,2) 2*C(2)*FV.normals(:,3) 2*C(2)*FV.normals(:,1) 2*C(1)*FV.normals(:,1).*FV.normals(:,2) 2*C(1)*FV.normals(:,2).*FV.normals(:,3) C(3)*FV.normals(:,3).*FV.normals(:,3)-C(5) 2*C(1)*FV.normals(:,1).*FV.normals(:,3) C(1)*(FV.normals(:,1).^2-FV.normals(:,2).^2)];
H = reshape(permute(repmat(H,[1 1 3]),[3 1 2]),[],size(H,2));
H(:,10:27)=0;
H(2:3:end,10:18)=H(2:3:end,1:9);
H(3:3:end,19:27)=H(3:3:end,1:9);
H(2:3:end,1:9)=0;
H(3:3:end,1:9)=0;

%% Generate Relighted Image 

tex_val = texPC(:,1:size(source.texb,1))*source.texb+texMU;
beta_i = repmat(tex_val,1,size(H,2)).*H;
Ir = reshape(beta_i*source.L,3,size(texPC,1)/3)';
Ir(Ir<0)=0;
FV.facevertexcdata = Ir;
Ir_src = renderFace(FV,source_img,FV.R,FV.t,FV.s,true);


tex_val = texPC(:,1:size(source.texb,1))*source.texb+texMU;
beta_i = repmat(tex_val,1,size(H,2)).*H;
Ir = reshape(beta_i*ref.L,3,size(texPC,1)/3)';
Ir(Ir<0)=0;
FV.facevertexcdata = Ir;
Ir_ref = renderFace(FV,source_img,FV.R,FV.t,FV.s,true);

out_img = (Ir_ref.*(source_img+0.01))./(Ir_src+0.01);

%% Results

figure(1);
subplot(131);imshow(source_img);title('source');
subplot(132);imshow(ref_img);title('reference');
subplot(133);imshow(out_img(:,:,1));title('Output');
