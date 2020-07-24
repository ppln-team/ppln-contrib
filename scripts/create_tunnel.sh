#!/usr/bin/env bash
# Helper for port forwarding
# based on https://unix.stackexchange.com/questions/100859/ssh-tunnel-without-shell-on-ssh-server

# Usage example:
# create_tunnel.sh \
#   --name=tunnel-name \
#   --local-addr=localhost \
#   --local-port=8991 \
#   --remote-addr=localhost \
#   --remote-port=9991 \
#   --remote-ssh=ssh-server-name \
#   --ssh-user=user-name \
#   --command=start
#
# create_tunnel.sh \
#   --n=tunnel-name \
#   -la=localhost \
#   -lp=8992 \
#   -ra=localhost \
#   -rp=9992 \
#   -rs=ssh-server-name \
#   -su=user-name \
#   -c=stop


set -e

# Args parsing
for i in "$@"; do
    case "$1" in
        -n=*|--name=*)
            FWD_NAME="${i#*=}"
            shift
        ;;
        -la=*|--local-addr=*)
            FWD_LOCAL_ADDR="${i#*=}"
            shift
        ;;
        -lp=*|--local-port=*)
            FWD_LOCAL_PORT="${i#*=}"
            shift
        ;;
        -ra=*|--remote-addr=*)
            FWD_REMOTE_ADDR="${i#*=}"
            shift
        ;;
        -rp=*|--remote-port=*)
            FWD_REMOTE_PORT="${i#*=}"
            shift
        ;;
        -rs=*|--remote-ssh=*)
            FWD_REMOTE_SSH="${i#*=}"
            shift
        ;;
        -su=*|--ssh-user=*)
            FWD_SSH_USER="${i#*=}"
            shift
        ;;
        -c=*|--command=*)
            CMD="${i#*=}"
            shift
        ;;
        *)
            echo "Invalid option $i"
            exit 1
    esac
done

FWD_SOCK_FILE=${FWD_SOCK_FILE:-"/tmp/pj-ssh-$FWD_NAME-fwd"}
FWD_LOCAL=$FWD_LOCAL_ADDR:$FWD_LOCAL_PORT
FWD_REMOTE=$FWD_REMOTE_ADDR:$FWD_REMOTE_PORT
echo "Tunneled $FWD_LOCAL <= $FWD_REMOTE (on $FWD_SSH_USER@$FWD_REMOTE_SSH)"


case $CMD in
    start)
        echo "Start forwarding ${FWD_NAME}: $FWD_REMOTE on $FWD_LOCAL"
        ssh -f -N -M -S "${FWD_SOCK_FILE}" -L "$FWD_LOCAL:$FWD_REMOTE" "${FWD_SSH_USER}@${FWD_REMOTE_SSH}"
        ;;
    stop)
        echo "Stop forwarding ($FWD_NAME) on :$FWD_LOCAL"
        ssh -S "${FWD_SOCK_FILE}" -O exit "${FWD_SSH_USER}@${FWD_REMOTE_SSH}"
        ;;
    *)
        echo "Error: unknown command $CMD"
        echo "Usage: ./bin/fwd.sh [start|stop]"
        exit 1
        ;;
esac

# TIPS ON ERRORS:
#
# Error: «channel_setup_fwd_listener_tcpip: cannot listen to port: »
# Sulution:
# lsof -i :$LOCAL_PORT
# kill $PID
