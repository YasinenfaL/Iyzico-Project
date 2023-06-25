# We give it to the directory where we have yaml and py files
cd mlops-final-project

# We will deploy with Kubernates  to create and run a local kubernates cluster
minikube start 

# To find out the status of the cluster
kubectl cluster-info 

#  To facilitate the integration between Docker and Minikube and in the development environment
# we used Docker to sync Minikube while developing app.
eval $(minikube docker-env)

# We will build a docker image in our local, it also contains dependencies
docker image build -t mlops-final:1.1 .   mlops-final: 1.1 etiket atadık

# We apply the yaml file we created
kubectl apply -f mlops-final-deployment.yaml


# We have created a service to open it to the outside world, let's check it out
kubectl create service nodeport mlops-final --tcp=8000:8000
kubectl get services


# we created nginx for interface

cat<<EOF | sudo tee /etc/nginx/nginx.conf
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log;
pid /run/nginx.pid;

include /usr/share/nginx/modules/*.conf;

events {
    worker_connections 1024;
}


http {
server {
        listen 80;
        server_name 127.0.0.1;

        location / {
                proxy_pass http://192.168.49.2:30963;
        }
}
}
EOF

# Let's check nginx status
sudo systemctl start nginx
sudo systemctl status nginx


# Activate Ingress
minikube addons enable ingress
sudo vim /etc/hosts
192.168.49.2    mlops-final.vbo.local

# Ingress+NGiNX Web UI
sudo vim /etc/nginx/nginx.conf

# Deployment
kubectl apply -f ingress-mlops-final.yaml




