/cannot/

/function/ { f = $8 }

/file/ { f = $8 }

/lines/ {

    k = match ($1, "%") ;
    p = substr ($1, 1, k-1) ;

    if ((p+0) != 100)
    {
        printf "%8s %s\n", p, f
    }
}

