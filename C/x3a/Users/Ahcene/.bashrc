# Lancer l'agent SSH automatiquement
eval "$(ssh-agent -s)" >/dev/null

# Ajouter automatiquement la clé privée
ssh-add ~/.ssh/id_ed25519 2>/dev/null

