clc
clear all
close all

%% Finding the Dominant Colours in an Image
I= imread('house.tiff');
X = reshape(I, 256*256, 3);
X = double(X);
R = [0,0,1];G = [0,1,0];B = [1,0,0]; 
figure, plot3(X(:,1), X(:,2), X(:,3),'.','Color',[1, 0, 0])


% Ilabeled = reshape(X(:,1), M, N, 3);
% figure, imshow( uint8( Ilabeled ) )

[cents, new_img, J, centroids_history] = k_means(X,2,[161,201,217;156,199,223],20);

figure(1)
plot(J); % track errors as plot
title('J(avg) vs epoch');
xlabel('epoch (# of Iterations)');
ylabel('J(avg)');
grid on

% Question (a) ii

figure (2)
plot3(centroids_history(:,1,1),centroids_history(:,2,1),centroids_history(:,3,1), '-*');
hold all
plot3(centroids_history(:,1,2),centroids_history(:,2,2),centroids_history(:,3,2), '-*');
grid on


% Question (a) iii
figure(3)
X1 = new_img(new_img(:,4)==1,:);
X2 = new_img(new_img(:,4)==2,:);
plot3(X1(:,1), X1(:,2), X1(:,3),'.','Color', cents(1,:)/256);
hold all
plot3(X2(:,1), X2(:,2), X2(:,3), '.', 'Color', cents(2,:)/256);
grid on

% Question (a) iv
figure(4)
redrawn_image = zeros(256,256,3);
counter = 0;
for row=1:256
    for column=1:256
        counter= counter+1;
        class_label = new_img(counter,4);
        redrawn_image(row,column,:) = cents(class_label,:);
    end
end
redrawn_image = uint8(redrawn_image);

redrawn_image = flip(imrotate(redrawn_image,-270,'bilinear','crop'));

subplot(1,2,1);imshow(I);
subplot(1,2,2);imshow(redrawn_image);
imwrite(redrawn_image,'k_2.png')

% Question (b)

[cents, new_img, J, centroids_history] = k_means(X,5,[161,201,217;156,199,223;158,102,90;159 94 91;96 53 78],20);
figure(5)
X1 = new_img(new_img(:,4)==1,:);
X2 = new_img(new_img(:,4)==2,:);
X3 = new_img(new_img(:,4)==3,:);
X4 = new_img(new_img(:,4)==4,:);
X5 = new_img(new_img(:,4)==5,:);
plot3(X1(:,1), X1(:,2), X1(:,3),'.','Color', cents(1,:)/256);
hold all
plot3(X2(:,1), X2(:,2), X2(:,3), '.', 'Color', cents(2,:)/256);
hold all
plot3(X3(:,1), X3(:,2), X3(:,3), '.', 'Color', cents(3,:)/256);
hold all
plot3(X4(:,1), X4(:,2), X4(:,3), '.', 'Color', cents(4,:)/256);
hold all
plot3(X5(:,1), X5(:,2), X5(:,3), '.', 'Color', cents(5,:)/256);
grid on
hold off

figure(6)
plot(J); % track errors as plot
title('J(avg) vs epoch');
xlabel('epoch (# of Iterations)');
ylabel('J(avg)');
grid on

[cents_2, new_img_2, J_2, centroids_history_2] = k_means(X,5,[157,0,222;157,198,222;157,198,221;101,82,98;96,101,124],20);
figure(7)
X1 = new_img_2(new_img_2(:,4)==1,:);
X2 = new_img_2(new_img_2(:,4)==2,:);
X3 = new_img_2(new_img_2(:,4)==3,:);
X4 = new_img_2(new_img_2(:,4)==4,:);
X5 = new_img_2(new_img_2(:,4)==5,:);
plot3(X1(:,1), X1(:,2), X1(:,3),'.','Color', cents_2(1,:)/256);
hold all
plot3(X2(:,1), X2(:,2), X2(:,3), '.', 'Color', cents_2(2,:)/256);
hold all
plot3(X3(:,1), X3(:,2), X3(:,3), '.', 'Color', cents_2(3,:)/256);
hold all
plot3(X4(:,1), X4(:,2), X4(:,3), '.', 'Color', cents_2(4,:)/256);
hold all
plot3(X5(:,1), X5(:,2), X5(:,3), '.', 'Color', cents_2(5,:)/256);
grid on
hold off

figure(8)
plot(J_2); % track errors as plot
title('J(avg) vs epoch');
xlabel('epoch (# of Iterations)');
ylabel('J(avg)');
grid on

figure(9)
redrawn_image2 = zeros(256,256,3);
counter = 0;
for row=1:256
    for column=1:256
        counter= counter+1;
        class_label = new_img(counter,4);
        redrawn_image2(row,column,:) = cents(class_label,:);
    end
end
redrawn_image2 = uint8(redrawn_image2);

redrawn_image2 = flip(imrotate(redrawn_image2,-270,'bilinear','crop'));

subplot(1,2,1);imshow(I);
subplot(1,2,2);imshow(redrawn_image2);
imwrite(redrawn_image2,'k_5.png')

% for i=1:20
%     %figure(i+6)
%     if (1<=i)&&(10>=i)
%         figure(7);
%         subplot(2, 5, i);
%     else
%         figure(8);
%         subplot(2, 5, i-10);
%     end
%     for j=1:5
%         X1 = new_img(new_img(:,4)==j,:);
%         plot3(X1(:,1), X1(:,2), X1(:,3),'.','Color', centroids_history(j,:,i)/256)
%         title("itr" +i);
%         hold all
%     end
% end
% grid on

%Question c)
N = size(X,1);
XB1 = 0;
for i=1:5
    Xi = new_img(new_img(:,4)==i,:);
    Xi = [Xi(:,1), Xi(:,2), Xi(:,3)];
    mu_i = new_img(new_img(:,4)~=i,:);
    mu_i = [mean(mu_i(:,1)), mean(mu_i(:,2)), mean(mu_i(:,3))];
    mu_j= repmat(cents(i,:),size(Xi,1),1);
    %var1=sum(sum((Xi-repmat(cents(i,:),5,1)).^2,2).^.5);
    var1 = (Xi-mu_j);
    var1 = sum((var1(:,1).^2 + var1(:,2).^2 + var1(:,3).^2).^.5)
    prime = repmat(cents(i,:),size(Xi,1),1);
    var2 = mu_i-prime;
    var2 = sum((var2(:,1).^2 + var2(:,2).^2 + var2(:,3).^2).^.5)
    XB1 = XB1 + var1\var2;
end
XB1 = XB1\N
