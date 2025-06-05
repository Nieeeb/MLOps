ssh $1@$2 'bash -s' <<'ENDSSH'
  # commands to run on remote host
  cd "$3" &&
  pwd;
ENDSSH
