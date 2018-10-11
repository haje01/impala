variable "access_key" {}
variable "secret_key" {}
variable "proj_owner" {}
variable "aws_key_name" {}
variable "region" {}
variable "availability_zone" {}
variable "ssh_cidr" {}
variable "ssh_key_file" {}
variable "num_task_node" {}
variable "actor_per_node" {}

output "subnet_cidr" {
    value = "${aws_default_subnet.default.cidr_block}"
}

output "master_ip" {
    value = "${aws_instance.master.public_ip}"
}

output "task_ips" {
    value = ["${aws_instance.task.*.public_ip}"]
}


provider "aws" {
  access_key = "${var.access_key}"
  secret_key = "${var.secret_key}"
  region = "${var.region}"
}


resource "aws_default_subnet" "default" {
    availability_zone = "${var.availability_zone}"

    tags {
        Name = "Default subnet"
    }
}


resource "aws_security_group" "distper" {
    name = "distper-sg"
    description = "Security group for DistPER."

    # for learner & buffer
    ingress {
        from_port = 5557
        to_port = 5558
        protocol = "tcp"
        cidr_blocks = ["${aws_default_subnet.default.cidr_block}"]
    }

    # for tensorboard
    ingress {
        from_port = 6006
        to_port = 6006
        protocol = "tcp"
        cidr_blocks = ["${var.ssh_cidr}"]
    }

    # for ssh
    ingress {
        from_port = 22
        to_port = 22
        protocol = "tcp"
        cidr_blocks = ["${var.ssh_cidr}"]
    }

    egress {
        from_port = 0
        to_port = 0
        protocol = "-1"
        cidr_blocks = ["0.0.0.0/0"]
    }

    tags {
        Name = "distper-sg"
        Owner = "${var.proj_owner}"
    }
}


# master node
resource "aws_instance" "master" {
    ami = "ami-b9e357d7"          # Deep Learning AMI (Ubuntu) Version 11.0
    instance_type = "p2.xlarge"   # GPU, 4 Cores, 61 GiB RAM
    key_name = "${var.aws_key_name}"
    vpc_security_group_ids = ["${aws_security_group.distper.id}"]
    subnet_id = "${aws_default_subnet.default.id}"

    provisioner "remote-exec" {
        connection {
            type = "ssh"
            user = "ubuntu"
            private_key = "${file("${var.ssh_key_file}")}"
        }
        inline = [
            <<EOF
sudo locale-gen ko_KR.UTF-8
/home/ubuntu/anaconda3/bin/conda install -n pytorch_p36 -y scikit-image tensorflow tensorboard opencv
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/pip install tensorboardX
/home/ubuntu/anaconda3/bin/conda install -n pytorch_p36 -y libprotobuf protobuf
git clone https://github.com/openai/gym
cd gym
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/pip install -e .
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/pip install gym[classic_control,atari]
cd
# git clone https://github.com/haje01/distper.git
git clone -b breakout https://github.com/haje01/distper.git  # FIXME
screen -S learner -dm bash -c "source anaconda3/bin/activate pytorch_p36; cd distper; python learner.py nowait; exec bash"
screen -S buffer -dm bash -c "source anaconda3/bin/activate pytorch_p36; cd distper; python buffer.py; exec bash"
screen -S board -dm bash -c "source anaconda3/bin/activate pytorch_p36; cd distper; tensorboard --logdir=runs; exec bash"
sleep 3
EOF
        ]
    }

    tags {
        Name = "distper-master"
        Owner = "${var.proj_owner}"
    }
}


# task node
resource "aws_instance" "task" {
    ami = "ami-b9e357d7"          # Deep Learning AMI (Ubuntu) Version 11.0
    instance_type = "m5.xlarge"
    key_name = "${var.aws_key_name}"
    vpc_security_group_ids = ["${aws_security_group.distper.id}"]
    subnet_id = "${aws_default_subnet.default.id}"
    count = "${var.num_task_node}"
    depends_on = ["aws_instance.master"]

    provisioner "remote-exec" {
        connection {
            type = "ssh"
            user = "ubuntu"
            private_key = "${file("${var.ssh_key_file}")}"
        }
        inline = [
            <<EOF
sudo locale-gen ko_KR.UTF-8
/home/ubuntu/anaconda3/bin/conda install -n pytorch_p36 -y opencv
/home/ubuntu/anaconda3/bin/conda install -n pytorch_p36 -y libprotobuf protobuf
git clone https://github.com/openai/gym
cd gym
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/pip install -e .
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/pip install gym[classic_control,atari]
cd
# git clone https://github.com/haje01/distper.git
git clone -b breakout https://github.com/haje01/distper.git  # FIXME
export MASTER_IP=${aws_instance.master.private_ip}
export TNODE_ID=${count.index}
export NUM_ACTOR=$((${var.num_task_node} * 4))
idx=0
while [ $idx -lt ${var.actor_per_node} ]
do
  screen -S "actor-$(($TNODE_ID*4+$idx))" -dm bash -c "source anaconda3/bin/activate pytorch_p36; cd distper; ACTOR_ID=$(($TNODE_ID*4+$idx)) python actor.py; exec bash"
  idx=`expr $idx + 1`
done
sleep 3
EOF
        ]
    }

    tags {
        Name = "distper-task"
        Owner = "${var.proj_owner}"
    }
}

