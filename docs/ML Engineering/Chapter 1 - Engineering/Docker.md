# Docker Cheatsheet

## Container Management

### Start/Stop/Restart

```bash
docker start <container>          # Start stopped container
docker stop <container>           # Graceful stop (SIGTERM)
docker kill <container>           # Force stop (SIGKILL)
docker restart <container>        # Restart container
```

### Run Containers

```bash
docker run -d <image>             # Run in background (detached)
docker run -it <image>            # Interactive with TTY
docker run -p 8080:80 <image>     # Port mapping
docker run -v /host:/container <image>  # Volume mount
docker run --rm <image>           # Auto-remove when stopped
docker run --name myapp <image>   # Custom name
```

### List & Inspect

```bash
docker ps                         # Running containers
docker ps -a                      # All containers (including stopped)
docker inspect <container>        # Detailed info
docker logs <container>           # View logs
docker logs -f <container>        # Follow logs
docker stats                      # Resource usage
```

### Execute Commands

```bash
docker exec -it <container> bash  # Interactive shell
docker exec <container> <command> # Run command
docker cp <container>:/path /host # Copy files from container
docker cp /host <container>:/path # Copy files to container
```

## Image Management

### Build & Tag

```bash
docker build -t <name>:<tag> .    # Build from Dockerfile
docker build -f <dockerfile> .    # Custom Dockerfile
docker tag <image> <new-name>     # Tag image
docker push <image>               # Push to registry
docker pull <image>               # Pull from registry
docker buildx build -f <dockerfile> -t <name>:<tag> --platform linux/amd64 . # build with specific platform
```

### List & Remove

```bash
docker images                     # List images
docker rmi <image>                # Remove image
docker rmi $(docker images -q)   # Remove all images
```

## Cleanup & Pruning

### Remove Containers

```bash
docker rm <container>             # Remove stopped container
docker rm -f <container>          # Force remove running container
docker rm $(docker ps -aq)       # Remove all containers
docker container prune           # Remove all stopped containers
```

### Remove Images

```bash
docker rmi <image>                # Remove image
docker image prune               # Remove dangling images
docker image prune -a            # Remove all unused images
```

### System Cleanup

```bash
docker system prune              # Remove stopped containers, dangling images, unused networks
docker system prune -a           # Remove all unused containers, images, networks, volumes (high pruning)
docker system prune --volumes    # Include volumes in cleanup
docker system df                 # Show disk usage
```

### Volume Management

```bash
docker volume ls                  # List volumes
docker volume rm <volume>        # Remove volume
docker volume prune              # Remove unused volumes
docker volume create <name>      # Create volume
```

### Network Management

```bash
docker network ls                # List networks
docker network rm <network>      # Remove network
docker network prune            # Remove unused networks
docker network create <name>     # Create network
```

## Bulk Operations

### Stop All Containers

```bash
docker stop $(docker ps -q)      # Stop all running containers
docker kill $(docker ps -q)      # Force stop all containers
```

### Remove All Containers

```bash
docker rm $(docker ps -aq)       # Remove all containers
docker rm -f $(docker ps -aq)    # Force remove all containers
```

### Remove All Images

```bash
docker rmi $(docker images -q)   # Remove all images
docker rmi -f $(docker images -q) # Force remove all images
```

## Docker Compose

### Basic Commands

```bash
docker-compose up                 # Start services
docker-compose up -d              # Start in background
docker-compose down               # Stop and remove containers
docker-compose stop               # Stop services
docker-compose restart            # Restart services
docker-compose build              # Build images
docker-compose pull               # Pull images
```

### Logs & Status

```bash
docker-compose logs               # View logs
docker-compose logs -f            # Follow logs
docker-compose ps                 # List services
docker-compose exec <service> bash # Execute command
```

## Registry Operations

### Login & Push

```bash
docker login                      # Login to Docker Hub
docker login <registry>           # Login to custom registry
docker push <image>               # Push image
docker pull <image>               # Pull image
docker search <term>              # Search Docker Hub
```

## Useful Flags & Options

### Common Flags

```bash
-d, --detach                      # Run in background
-it                               # Interactive + TTY
-p, --publish                     # Port mapping
-v, --volume                      # Volume mount
--rm                              # Auto-remove container
--name                            # Container name
-e, --env                         # Environment variable
--network                         # Network to connect
--restart                         # Restart policy
```

### Resource Limits

```bash
--memory 512m                     # Memory limit
--cpus 1.5                        # CPU limit
--memory-swap 1g                  # Swap limit
```

## Troubleshooting

### Debug Commands

```bash
docker inspect <container>        # Detailed container info
docker logs --details <container> # Detailed logs
docker events                     # Real-time events
docker version                    # Docker version
docker info                       # System info
```

### Common Issues

```bash
# Permission denied
sudo docker <command>

# Port already in use
docker ps | grep <port>

# Out of disk space
docker system prune -a --volumes

# Container won't stop
docker kill <container>
```

## Quick Cleanup Script

```bash
#!/bin/bash
# Nuclear cleanup - removes everything
docker kill $(docker ps -q) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null
docker rmi $(docker images -q) 2>/dev/null
docker volume prune -f
docker network prune -f
docker system prune -af --volumes
```
