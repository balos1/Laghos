NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr qdeim restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -offline -writesol -romsvds -no-romsns -romos
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 12 -rdime 6 -soldiff -no-romsns -romos
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 12 -rdime 6 -rdimfv 16 -rdimfe 9 -romhrprep -nsamx 4 -nsamv 24 -nsame 32 -no-romsns -romos
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 12 -rdime 6 -rdimfv 16 -rdimfe 9 -romhr -nsamx 4 -nsamv 24 -nsame 32 -no-romsns -romos
    ;;
  4)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 12 -rdime 6 -rdimfv 16 -rdimfe 9 -romhrprep -qdeim -no-romsns -romos
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 12 -rdime 6 -rdimfv 16 -rdimfe 9 -romhr -qdeim -no-romsns -romos
    ;;
  5)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -restore -rdimx 2 -rdimv 12 -rdime 6 -soldiff -no-romsns -romos
    ;;
esac