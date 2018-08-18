function [centroids,img_new,J_p,centroids_history] = k_means(img,k,ic,epoch_limit)
%ic is the matrix containing the initial centroids values
%k is the number of classes we expect
centroids_history = zeros(k,3,epoch_limit);
centroids = ic;
new_centroids = zeros( size(centroids(:,:,1)) );
img_new = zeros(size(img,1),4);
epoch = 0;
flag_continue = 1;
J_p = zeros(1,epoch_limit);
while (flag_continue == 1 && epoch < epoch_limit)
    epoch = epoch + 1; 
    for i=1:size(img,1)
        pixel = img(i,:);
        distances = zeros(1,k);
        for centroid_index=1:k
            curr_centroid = centroids(centroid_index,:);
            distance = ((pixel(1)-curr_centroid(1))^2 + (pixel(2)-curr_centroid(2))^2 + (pixel(3)-curr_centroid(3))^2);
            distances(centroid_index) = distance;
            J_p(epoch) = J_p(epoch) + distance;
        end
        [~, min_index] = min(distances);
        img_new(i,:) = [pixel min_index]; % chooses the index of the smallest distance and labels the pixel according to the index 

    end
    for i=1:k
        grouped_pixels= img_new(img_new(:,4)==i,:);
        R = mean(grouped_pixels(:,1));
        G = mean(grouped_pixels(:,2));
        B = mean(grouped_pixels(:,3));
        new_centroids(i,:) = [R,G,B];
    end
    if (new_centroids == centroids)
        flag_continue = 0;
    end 
    centroids = new_centroids;
    centroids_history(:,:,epoch) = centroids;
end

end

