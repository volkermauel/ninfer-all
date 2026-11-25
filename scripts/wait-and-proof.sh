#!/usr/bin/env bash
# Fires the paging proof when GPU 1 shows <4 GiB used for 60 s straight.
while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    if [ "$USED" -lt 4000 ]; then
        OK=1
        for i in 1 2 3; do
            sleep 20
            USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
            [ "$USED" -lt 4000 ] || { OK=0; break; }
        done
        [ "$OK" = 1 ] && break
    fi
    sleep 30
done
echo "$(date -Is) GPU 1 free (${USED} MiB) — starting proof"
exec bash ~/git/ninfer-all-paged/scripts/proof-paged-kv.sh
