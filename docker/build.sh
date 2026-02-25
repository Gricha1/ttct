if [ -z "$1" ]; then
    postfix=
else
    postfix=_$1
fi

cd ..
# check if safety-gym exist
docker build -t ttct_img -f docker/dockerfile .