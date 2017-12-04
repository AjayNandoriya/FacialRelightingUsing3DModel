data = csvread('/home/qcri/Documents/ajay/CNN/Face/scripts/comb_1508_testing1_different_target.csv');
[pid,pidname] = textread('/home/qcri/Documents/ajay/CNN/Face/dataset/pid_list.txt','%d %s');
model3d_dir  = '/home/qcri/Documents/ajay/CNN/Face/SOA/3DMM_edges/mean_img_rotate';
coeff_dir = '/home/qcri/Documents/ajay/CNN/Face/SOA/3DMM_edges/reconstruct/coeff';
outdir ='results/in_full';

addpath(genpath('/home/qcri/Documents/ajay/CNN/Face/SOA/3DMM_edges'));

% YOU MUST set this to the base directory of the Basel Face Model
BFMbasedir = '/home/qcri/Documents/ajay/CNN/Face/SOA/3dmm_cnn/3DMM_model/';

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

for k=1:length(data)
    
    pid_val = data(k,1);
    lid_in = data(k,2);
    lid_out = data(k,3);
    
    outfname = sprintf('%s/%04d_%03d_%03d_output.png',outdir,k,pid_val,lid_out);
    
    maskfname = sprintf('%s/%04d_%03d_%03d_mask.png',outdir,k,pid_val,lid_out);
    if(~exist(maskfname,'file') )
        fname= fullfile('smb://desktop-v9v7nvv/users/qcri/Documents/Ajay/Face/SOA/comparison/ours_baseline_recuurent_darkness100/images',sprintf('masks_%03d.png',pid_val));
        mimg = imresize(imread(fname),[1300 1030]);
        imwrite(mimg,maskfname);
    end
    continue;
    
    if(exist(outfname,'file') )
        continue;
    else
        fprintf('processing %s \n',outfname);
    end
    
    if(pid_val==218)
        img = im2double(imread(['/media/NAS/Dropbox_MIT/MERL_facial/' sprintf('%s/refl1_%03d_15.png',pidname{pid_val+1},lid_in)]));
        img = permute(img,[2 1 3]);
        img_gt = im2double(imread(['/media/NAS/Dropbox_MIT/MERL_facial/' sprintf('%s/refl1_%03d_15.png',pidname{pid_val+1},lid_out)]));
        img_gt = permute(img_gt,[2 1 3]);
        imwrite(img,outfname);
        imwrite(img_gt,sprintf('%s/%04d_%03d_%03d_gt.png',outdir,k,pid_val,lid_out));
        continue;
    end
    load(['/home/qcri/Documents/ajay/CNN/Face/SOA/3DMM_edges/mean_img_rotate/mean_' sprintf('%03d',pid_val) '_15.mat']);
    % b = csvread('/home/qcri/Documents/ajay/CNN/Face/SOA/3dmm_cnn/demoCode/mean_out/mean_004_1_15.ply.alpha');
    % b=b(1:ndims);
    FV.vertices=reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC,1)/3)';
    FV.faces = tl;
    pts2D = pts3Dto2D(FV.vertices,R,t,s)';

    FV.visiblePoints = visiblevertices(FV,R);
    FV.normals = vertex_normals(FV,R,t,s);
    % img = im2double(imread('/home/qcri/Documents/ajay/CNN/Face/dataset/mean_img_rotate/mean_004_1_15.png'));
    img = im2double(imread(['/media/NAS/Dropbox_MIT/MERL_facial/' sprintf('%s/refl1_%03d_15.png',pidname{pid_val+1},lid_in)]));
    img = permute(img,[2 1 3]);
    img_gt = im2double(imread(['/media/NAS/Dropbox_MIT/MERL_facial/' sprintf('%s/refl1_%03d_15.png',pidname{pid_val+1},lid_out)]));
    img_gt = permute(img_gt,[2 1 3]);
    C = [0.429043 0.511664 0.743125 0.886227 0.247708];
    H = [C(4)*ones(length(FV.normals),1) 2*C(2)*FV.normals(:,2) 2*C(2)*FV.normals(:,3) 2*C(2)*FV.normals(:,1) 2*C(1)*FV.normals(:,1).*FV.normals(:,2) 2*C(1)*FV.normals(:,2).*FV.normals(:,3) C(3)*FV.normals(:,3).*FV.normals(:,3)-C(5) 2*C(1)*FV.normals(:,1).*FV.normals(:,3) C(1)*(FV.normals(:,1).^2-FV.normals(:,2).^2)];
    H = reshape(permute(repmat(H,[1 1 3]),[3 1 2]),[],size(H,2));
    H(:,10:27)=0;
    H(2:3:end,10:18)=H(2:3:end,1:9);
    H(3:3:end,19:27)=H(3:3:end,1:9);
    H(2:3:end,1:9)=0;
    H(3:3:end,1:9)=0;
    
    cond1 = load(fullfile(coeff_dir,sprintf('coeff_%03d_%03d_15.mat',pid_val,lid_in)));
    cond2 = load(fullfile(coeff_dir,sprintf('coeff_%03d_%03d_15.mat',pid_val,lid_out)));
    tex_val = texPC(:,1:ndims)*cond1.texb+texMU;
    beta_i = tex_val.*H;
    Ir = reshape(beta_i*cond1.L,3,size(texPC,1)/3)';
    Ir(Ir<0)=0;
    FV.facevertexcdata = Ir;
    Ir1 = renderFace(FV,img,R,t,s,true);
    
    tex_val = texPC(:,1:ndims)*cond2.texb+texMU;
    beta_i = tex_val.*H;
    Ir = reshape(beta_i*cond2.L,3,size(texPC,1)/3)';
    Ir(Ir<0)=0;
    FV.facevertexcdata = Ir;
    Ir2 = renderFace(FV,img,R,t,s,true);
    
    
    Ir1 = imfilter(Ir1,fspecial('gaussian',7,3),'symmetric');
    Ir2 = imfilter(Ir2,fspecial('gaussian',7,3),'symmetric');
    
    I_out = (Ir2.*(img+0.01))./(Ir1+0.01);
    imwrite(I_out,outfname);
    imwrite(img_gt,sprintf('%s/%04d_%03d_%03d_gt.png',outdir,k,pid_val,lid_out));
end