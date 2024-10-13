while getopts g:n:p: flag
do
        case "${flag}" in
                g)  gpu=${OPTARG};;
                n)  number=${OPTARG};;
                p)  port=${OPTARG};;
        esac
done
echo "Running container whispervits-svc-conversion$number on gpu $gpu and port $port";

docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=$gpu --runtime=nvidia --userns=host --shm-size 64G -v /work/leonardo.boulitreau/whispervits-svc:/workdir/whispervits-svc/  -p $port --name whispervits-svc$number gg-voice-conversion:latest /bin/bash
