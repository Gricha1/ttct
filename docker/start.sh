if [ -z "$1" ]; then
    device=0
else
    device=$1
fi

if [ -z "$2" ]; then
    docker_container_idx=0
else
    docker_container_idx=$2
fi

if [ -z "$3" ]; then
    image_name=ttct_img
else
    image_name=$3
fi

echo "start dockergpu device: $device"
echo "start docker name: hrac_$docker_container_idx"
echo "start docker image: $image_name"

cd ..
docker run -it --rm --name ttct_$docker_container_idx --gpus "device=$device" --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $(pwd):/usr/home/workspace $image_name "bash"