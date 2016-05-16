-- noinspection SqlResolveForFile
-- noinspection SqlNoDataSourceInspectionForFile
use EXPORT_PATRICKN
go

declare @ratingTemp table(userid int, adid int, rating float)
declare @uTemp table(id int, idx int)
declare @aTemp table(id int, idx int)

-- get sample data
insert into @ratingTemp(userid, adid, rating)
select top 10000 r.UserId, r.AdID, r.Rating
from Rating r
order by r.AdID

insert into @uTemp(id, idx)
select t.userid, ROW_NUMBER() OVER(ORDER BY t.userid) as uIndex
from @ratingTemp t group by t.userid

insert into @aTemp(id, idx)
select t.adid, ROW_NUMBER() OVER(ORDER BY t.adid) as aIndex
from @ratingTemp t group by t.adid

-- compare count of users vs max row indexes
select COUNT(distinct t.userid) as [count] from @ratingTemp t
union all
select MAX(idx) from @uTemp
union all
select COUNT(distinct t.adid) from @ratingTemp t
union all
select MAX(idx) from @aTemp

-- get count of ratings
select COUNT(1)
from(
  select u.idx-1 as uIndex, a.idx-1 as aIndex
  from @ratingTemp t
    inner join @uTemp u on u.id = t.userid
    inner join @aTemp a on a.id = t.adid
  where t.rating != 0
  group by u.idx-1, a.idx-1 ) t

/*
get data in coordinate form
- uIndex from 0 to [n-1] (number of users)
- aIndex from 0 to [m-1] (number of ads)
*/
select t.userid, u.idx-1 as uIndex, t.adid, a.idx-1 as aIndex, t.Rating
from @ratingTemp t
	inner join @uTemp u on u.id = t.userid
	inner join @aTemp a on a.id = t.adid
order by adid, userid

-- get all ads
select distinct a.[AdID]
      ,[CreateDate]
      ,[ChangeDate]
      ,[AdStatusID]
      ,[SiteID]
      ,[AdGroupID]
      ,[ParentAdID]
      ,[PropertyTypeID]
      ,[DealTypeID]
      ,[AvailableDate]
      ,[HasCableTV]
      ,[HasCellar]
      ,[HasConservatory]
      ,[HasDishwasher]
      ,[HasDryer]
      ,[HasFireplace]
      ,[HasGarden]
      ,[HasGardenhouse]
      ,[HasLift]
      ,[HasPlayground]
      ,[HasTinkerroom]
      ,[HasWashingmachine]
      ,[HasWoodStove]
      ,[IsAttic]
      ,[IsBrightly]
      ,[IsBuildingLandConnected]
      ,[IsCentral]
      ,[IsChildFriendly]
      ,[IsCornerHouse]
      ,[IsFirstTimeUse]
      ,[IsFlatShare]
      ,[IsFurnished]
      ,[IsMiddleHouse]
      ,[IsMinergieCertified]
      ,[IsNewBuilding]
      ,[IsOldBuilding]
      ,[IsQuiet]
      ,[IsRaisedGroundFloor]
      ,[IsSunny]
      ,[IsUnderBuildingLaws]
      ,[IsUnderRoof]
      ,[IsWheelchairAccessable]
      ,[NumApartments]
      ,[NumBalconies]
      ,[NumBaths]
      ,[NumBathtubs]
      ,[NumFloors]
      ,[NumParkingIndoor]
      ,[NumParkingOutdoor]
      ,[NumPorches]
      ,[NumRooms]
      ,[NumShowers]
      ,[NumSwimmingPools]
      ,[NumTerraces]
      ,[NumToilets]
      ,[YearOfConstruction]
      ,[CountryID]
      ,[Zip]
      ,[City]
      ,[Street]
      ,[Condition]
      ,[CurrencyID]
      ,[DistanceBus]
      ,[DistanceHighway]
      ,[DistanceKindergarden]
      ,[DistancePlayground]
      ,[DistancePrimarySchool]
      ,[DistanceSecondarySchool]
      ,[DistanceShoppingFacilities]
      ,[DistanceTimeBus]
      ,[DistanceTimeHighway]
      ,[DistanceTimeShoppingFacilities]
      ,[DistanceTrainStation]
      ,[DistanceUniversity]
      ,[Floor]
      ,[Flooring]
      ,[LivingSpace]
      ,[BuildingArea]
      ,[BalconyMeters]
      ,[GardenMeters]
      ,[Heating]
      ,[LastModifiedOnSite]
      ,[MinergieGeneral]
      ,[GrossRent]
      ,[NetRent]
      ,[SideCost]
      ,[PurchasePrice]
      ,[Orientation]
      ,[ParkingIndoorText]
      ,[ParkingOutdoorText]
      ,[PetsAllowed]
      ,[PhoneType]
      ,[RenovationYear]
      ,[Situation]
      ,[StoveType]
      ,[ViewText]
      ,[Volume]
      ,[VendorCity]
      ,[VendorCountry]
      ,[VendorPriority]
      ,[VendorType]
      ,[VendorWebsite]
      ,[VendorZIP]
      ,[FlatRate]
      ,[PriceModel]
      ,[LastCreateMessageID]
      ,[LastUpdateMessageID]
      ,[LastDeleteMessageID]
      ,[LastMutatedMessageID]
      ,[GeoPosQuality]
      ,[GeoDataSource]
      ,[LevenshteinMatch]
      ,[LevenshteinFactor]
      ,[GeoPosX]
      ,[GeoPosY]
      ,[GeoPosLng]
      ,[GeoPosLat]
      ,[GeoSystem]
      ,[District]
      ,[StreetNumber]
      ,[ComparisPoints]
      ,[ComparisPrice]
      ,[ComparisPriceMin]
      ,[ComparisPriceMax]
      ,[CommunityID]
      ,[CommunityName]
      ,[UsefulArea]
      ,[PriceTypeID]
      ,[DuplicateGroupID]
      ,[LastPriceChangeDate]
      ,[LastPriceChange]
      ,[OverallPriceChange]
      ,[LocationID]
      ,[PartialMatch]
      ,[RealEstateID]
      ,[DeleteDate]
      ,[SiteIDOriginal]
     from @ratingTemp t
	inner join Ad a on a.AdID = t.adid