pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract KoiFarmFactory is Ownable {
    using SafeMath for uint256;

    struct Farm {
        address lpToken;
        address rewardToken;
        uint256 totalAllocPoint;
        uint256 lastRewardBlock;
        uint256 accumulatedRewardPerToken;
    }

    IERC20 public koiToken;
    mapping(address => Farm) public farms;
    mapping(address => uint256) public allocPoint;
    uint256 private _koiPerBlock;

    constructor(address _koiToken, uint256 _koiPerBlock) {
        koiToken = IERC20(_koiToken);
        _koiPerBlock =_koiPerBlock;
        allocPoint[msg.sender] = 1000;
    }

    function createFarm(address _lpToken, address _rewardToken) public {
        require(msg.sender == owner, "Only the owner can create a farm");
        require(farms[msg.sender].lpToken == address(0), "The owner already has a farm");
        require(_lpToken != address(0), "Invalid LP token");
        require(_rewardToken != address(0), "Invalid reward token");

        Farm memory newFarm = Farm(_lpToken, _rewardToken, 0, block.timestamp, 0);
        farms[msg.sender] = newFarm;
        allocPoint[msg.sender] = 100;
    }

    function setKoiPerBlock(uint256 _koiPerBlock) public onlyOwner {
        _koiPerBlock = _koiPerBlock;
    }

    function updateFarm(address _user, uint256 _allocPoint) public onlyOwner {
        require(farms[_user].lpToken != address(0), "The user does not have a farm");
        require(_allocPoint > 0 && _allocPoint <= 10000, "Invalid allocation point");

        allocPoint[_user] = _allocPoint;
    }

    function notifyRewardAmount(uint256 _amount) external {
        uint256 koiReward = _amount.mul(_koiPerBlock).div(1e18);
        uint256 totalAllocPoint = 0;
        for (address user : getUsers()) {
            totalAllocPoint = totalAllocPoint.add(allocPoint[user]);
        }

        for (address user : getUsers()) {
            Farm storage farm = farms[user];
            uint256 userReward = koiReward.mul(allocPoint[user]).div(totalAllocPoint);
            farm.accumulatedRewardPerToken = farm.accumulatedRewardPerToken.add(userReward);
        }
    }

    function getUsers() public view returns (address[] memory) {
        uint256 length = 0;
        for (uint256 i = 0; i < allocPoint.length; i++) {
            if (allocPoint[address(uint160(i))] > 0) {
                length++;
            }
        }

        address[] memory users = new address[](length);
        uint256 index = 0;
        for (uint256 i = 0; i < allocPoint.length; i++) {
            if (allocPoint[address(uint160(i))] > 0) {
                users[index] = address(uint160(i));
                index++;
            }
        }

        return users;
    }

    function getUserFarm(address _user) public view returns (address, address, uint256, uint256, uint256) {
        Farm memory farm = farms[_user];
        return (farm.lpToken, farm.rewardToken, farm.totalAllocPoint, farm.lastRewardBlock, farm.accumulatedRewardPerToken);
    }

    function deposit(address _lpToken, uint256 _amount) external {
        Farm storage farm = farms[msg.sender];
        require(farm.lpToken == _lpToken, "Invalid LP token");

        IERC20(_lpToken).transferFrom(msg.sender, address(this), _amount);
        koiToken.mint(msg.sender, _amount);

        farm.totalAllocPoint = farm.totalAllocPoint.add(1);
    }

    function withdraw(address _lpToken, uint256 _amount) external {
        Farm storage farm = farms[msg.sender];
        require(farm.lpToken == _lpToken, "Invalid LP token");

        IERC20(_lpToken).transferFrom(address(this), msg.sender, _amount);
        koiToken.burn(msg.sender, _amount);

        farm.totalAllocPoint = farm.totalAllocPoint.sub(1);
    }

    function emergencyWithdraw(address _lpToken) external {
        Farm storage farm = farms[msg.sender];
        require(farm.lpToken == _lpToken, "Invalid LP token");

        IERC20(_lpToken).transferFrom(address(this), msg.sender, IERC20(_lpToken).balanceOf(address(this)));
        koiToken.burn(msg.sender, IERC20(_lpToken).balanceOf(address(this)));

        farm.totalAllocPoint = 0;
    }

    function withdrawReward(address _lpToken) external {
        Farm storage farm = farms[msg.sender];
        require(farm.lpToken == _lpToken, "Invalid LP token");

        uint256 reward = farm.accumulatedRewardPerToken.mul(IERC20(_lpToken).balanceOf(address(this))).div(1e18);
        koiToken.transfer(msg.sender, reward);

        farm.accumulatedRewardPerToken = 0;
    }
}